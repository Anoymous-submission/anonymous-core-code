from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wan_video_dit import flash_attention, modulate, rope_apply
from fasterwam.utils.logging_config import get_logger

logger = get_logger(__name__)


class MoT(nn.Module):
    def __init__(
        self,
        mixtures: Dict[str, nn.Module],
        mot_checkpoint_mixed_attn: bool = True,
        mixed_attention_type: str = "softmax",
    ):
        super().__init__()
        if not mixtures:
            raise ValueError("`mixtures` cannot be empty.")
        if "video" not in mixtures or "action" not in mixtures:
            raise ValueError("`mixtures` must include both 'video' and 'action' experts.")

        self.mixtures = nn.ModuleDict(mixtures)
        self.expert_order = list(self.mixtures.keys())
        self.mot_checkpoint_mixed_attn = mot_checkpoint_mixed_attn
        self.mixed_attention_type = str(mixed_attention_type)
        if self.mixed_attention_type not in {"softmax", "linear"}:
            raise ValueError(
                "mixed_attention_type must be 'softmax' or 'linear', got "
                f"{self.mixed_attention_type!r}"
            )
        if mot_checkpoint_mixed_attn:
            logger.info(
                "Using gradient checkpointing for mixture attention. This will save memory but use more computation."
            )

        first_expert = self.mixtures[self.expert_order[0]]
        self.num_layers = len(first_expert.blocks)
        self.num_heads = first_expert.num_heads
        self.attn_head_dim = first_expert.attn_head_dim

        for name in self.expert_order[1:]:
            expert = self.mixtures[name]
            if len(expert.blocks) != self.num_layers:
                raise ValueError(
                    f"All experts must have same number of layers; got {self.num_layers} and {len(expert.blocks)}"
                )
            if expert.num_heads != self.num_heads:
                raise ValueError(
                    f"All experts must have same num_heads; got {self.num_heads} and {expert.num_heads}"
                )
            if expert.attn_head_dim != self.attn_head_dim:
                raise ValueError(
                    "All experts must have same attn_head_dim; "
                    f"got {self.attn_head_dim} and {expert.attn_head_dim}"
                )

        logger.info(
            f"Initialized MoT with experts: {self.expert_order}, num_layers={self.num_layers}"
        )
        for name in self.expert_order:
            expert = self.mixtures[name]
            logger.info(
                f"  Expert '{name}': num_params={sum(p.numel() for p in expert.parameters()) / 1e9:.2f} B"
            )

    @staticmethod
    def _split_modulation(block, t_mod: torch.Tensor):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1

        base_mod = block.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (base_mod + t_mod).chunk(
            6, dim=chunk_dim
        )
        if has_seq:
            # means t_mod has separate modulation for each token, otherwise same modulation for all tokens in the block
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def _mixed_attention(
        self,
        q_cat: torch.Tensor,
        k_cat: torch.Tensor,
        v_cat: torch.Tensor,
        attention_mask: torch.Tensor,
        linear_attention_groups: Optional[Dict[str, torch.Tensor]] = None,
        linear_q_den: Optional[torch.Tensor] = None,
        linear_k_den: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_mask = attention_mask.to(device=q_cat.device)

        def _forward(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            q_den: torch.Tensor,
            k_den: torch.Tensor,
        ) -> torch.Tensor:
            if self.mixed_attention_type == "linear":
                if linear_attention_groups is None:
                    raise ValueError(
                        "linear mixed attention requires explicit condition/prediction/missing groups"
                    )
                return self._linear_mixed_attention(q, k, v, q_den, k_den, linear_attention_groups)
            return flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=attn_mask)

        if self.mixed_attention_type == "linear":
            if linear_q_den is None or linear_k_den is None:
                raise ValueError("linear attention requires unrotated denominator features")
            q_den = linear_q_den
            k_den = linear_k_den
        else:
            # These arguments are ignored by the softmax branch, but passing
            # tensors keeps the checkpoint closure explicit and gradient-safe.
            q_den = q_cat
            k_den = k_cat

        if self.mot_checkpoint_mixed_attn and self.training:
            return torch.utils.checkpoint.checkpoint(
                _forward,
                q_cat,
                k_cat,
                v_cat,
                q_den,
                k_den,
                use_reentrant=False,
            )
        return _forward(q_cat, k_cat, v_cat, q_den, k_den)

    def _linear_mixed_attention(
        self,
        q_cat: torch.Tensor,
        k_cat: torch.Tensor,
        v_cat: torch.Tensor,
        q_den_cat: torch.Tensor,
        k_den_cat: torch.Tensor,
        groups: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Normalized ELU+1 linear attention with the H&R visibility contract.

        Valid condition queries read valid condition keys. Prediction queries
        read valid conditions plus every prediction. Missing-condition queries
        read only themselves and are never exposed as keys to another token.
        The sufficient statistics are accumulated in FP32 for stability, then
        cast back to the input dtype. This is O(S * Dh^2), not O(S^2 * Dh).
        """

        batch, seq_len, packed = q_cat.shape
        if packed != self.num_heads * self.attn_head_dim:
            raise ValueError(
                f"packed attention dim {packed} != heads*head_dim "
                f"{self.num_heads * self.attn_head_dim}"
            )
        required = {"condition", "prediction", "missing"}
        if set(groups) != required:
            raise ValueError(
                f"linear_attention_groups must have exactly {sorted(required)}, got {sorted(groups)}"
            )
        normalized = {}
        for name in required:
            value = groups[name].to(device=q_cat.device, dtype=torch.bool)
            if value.shape != (batch, seq_len):
                raise ValueError(
                    f"linear_attention_groups[{name!r}] must be [B,S] "
                    f"= {(batch, seq_len)}, got {tuple(value.shape)}"
                )
            normalized[name] = value
        membership = sum(value.to(torch.int8) for value in normalized.values())
        if not bool((membership == 1).all()):
            raise ValueError("linear attention groups must partition every token exactly once")

        q_num = q_cat.reshape(batch, seq_len, self.num_heads, self.attn_head_dim)
        k_num = k_cat.reshape(batch, seq_len, self.num_heads, self.attn_head_dim)
        q_den = q_den_cat.reshape(batch, seq_len, self.num_heads, self.attn_head_dim)
        k_den = k_den_cat.reshape(batch, seq_len, self.num_heads, self.attn_head_dim)

        def pooled(pool: torch.Tensor) -> torch.Tensor:
            weight = pool[:, :, None, None].to(k_num.dtype)
            selected_k_num = k_num * weight
            selected_k_den = k_den * weight
            kv = torch.einsum("bshd,bshe->bhde", selected_k_num, v_float)
            k_sum = selected_k_den.sum(dim=1)
            numerator = torch.einsum("bshd,bhde->bshe", q_num, kv)
            denominator = torch.einsum("bshd,bhd->bsh", q_den, k_sum).unsqueeze(-1)
            return numerator / denominator.clamp_min(1e-6)

        # CUDA autocast would otherwise silently execute einsum in BF16 even
        # when its inputs were explicitly converted to float.  The sufficient
        # statistics and normalization are intentionally FP32.
        with torch.autocast(device_type=q_cat.device.type, enabled=False):
            q_num = q_num.float()
            k_num = k_num.float()
            q_den = q_den.float()
            k_den = k_den.float()
            v_float = v_cat.reshape(batch, seq_len, self.num_heads, self.attn_head_dim).float()
            condition_out = pooled(normalized["condition"])
            readable = normalized["condition"] | normalized["prediction"]
            prediction_out = pooled(readable)
            # A missing condition has a diagonal-only row in the canonical
            # mask, so its output is exactly its own value.
            output = torch.where(normalized["condition"][:, :, None, None], condition_out, v_float)
            output = torch.where(normalized["prediction"][:, :, None, None], prediction_out, output)
        return output.to(q_cat.dtype).reshape(batch, seq_len, packed)

    def _resolve_attention_masks(
        self,
        attention_mask: torch.Tensor | Sequence[torch.Tensor],
        expected_seq_len: int,
    ) -> list[torch.Tensor]:
        if isinstance(attention_mask, torch.Tensor):
            attention_masks = [attention_mask] * self.num_layers
        else:
            attention_masks = list(attention_mask)
            if len(attention_masks) != self.num_layers:
                raise ValueError(
                    f"`attention_mask` must contain {self.num_layers} layers, got {len(attention_masks)}."
                )

        for layer_idx, layer_mask in enumerate(attention_masks):
            if not isinstance(layer_mask, torch.Tensor):
                raise ValueError(
                    f"`attention_mask[{layer_idx}]` must be torch.Tensor, got {type(layer_mask)}."
                )
            # [S,S] shares one mask across the batch; [B,S,S] and [B,1,S,S] give a
            # per-sample mask, which is what a per-sample condition-availability
            # pattern needs. All three broadcast correctly in
            # scaled_dot_product_attention against q of shape [B,heads,S,D].
            if layer_mask.ndim not in (2, 3, 4):
                raise ValueError(
                    f"`attention_mask[{layer_idx}]` must be [S,S], [B,S,S] or [B,1,S,S], "
                    f"got shape {tuple(layer_mask.shape)}"
                )
            if layer_mask.shape[-1] != layer_mask.shape[-2]:
                raise ValueError(
                    f"`attention_mask[{layer_idx}]` must be square in its last two dims, "
                    f"got shape {tuple(layer_mask.shape)}"
                )
            if layer_mask.shape[-1] != expected_seq_len:
                raise ValueError(
                    f"`attention_mask[{layer_idx}]` seq length mismatch: "
                    f"mask={layer_mask.shape[-1]} vs expected_total={expected_seq_len}"
                )
        return attention_masks

    @staticmethod
    def _apply_expert_post_block(
        block,
        residual_x: torch.Tensor,
        mixed_attn_out: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        x = block.gate(residual_x, gate_msa, block.self_attn.o(mixed_attn_out))

        if context_payload is not None:
            context = context_payload.get("context")
            if context is not None:
                context_mask = context_payload.get("mask")
                if context_mask is not None and context_mask.dim() == 3:
                    context_mask = context_mask.unsqueeze(1)
                x = x + block.cross_attn(block.norm3(x), context, ctx_mask=context_mask)

        mlp_input = modulate(block.norm2(x), shift_mlp, scale_mlp)
        x = block.gate(x, gate_mlp, block.ffn(mlp_input))
        return x

    def _build_expert_attention_io(
        self,
        expert,
        block,
        x: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        bool,
    ]:
        """Build per-expert attention tensors and post-block states.

        Args:
            expert: Expert module that owns this `block`; only used to read
                `use_gradient_checkpointing`.
            block: Transformer block for current layer (`expert.blocks[layer_idx]`).
            x: Current expert tokens, shape [B, S, D].
            freqs: RoPE frequencies aligned with token sequence, shape [S, 1, rope_dim].
            t_mod: Time modulation tensor for this expert/layer.

        Returns:
            q: Query used by the attention numerator, shape [B, S, H*Dh].
            k: Key used by the attention numerator, shape [B, S, H*Dh].
            v: Value after v-proj, shape [B, S, H*Dh].
            q_den: Query used by the linear-attention denominator.
            k_den: Key used by the linear-attention denominator.
            residual_x: Original input `x` for residual path in post block.
            gate_msa: Gating tensor for self-attention residual branch.
            shift_mlp: Shift tensor for MLP modulation.
            scale_mlp: Scale tensor for MLP modulation.
            gate_mlp: Gating tensor for MLP residual branch.
            use_gradient_checkpointing: Whether this expert enables checkpointing.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._split_modulation(
            block, t_mod
        )
        attn_input = modulate(block.norm1(x), shift_msa, scale_msa)

        q = block.self_attn.norm_q(block.self_attn.q(attn_input))
        k = block.self_attn.norm_k(block.self_attn.k(attn_input))
        v = block.self_attn.v(attn_input)

        if self.mixed_attention_type == "linear":
            # RoFormer-compatible linear RoPE: rotate the positive feature map
            # in the numerator, but keep its unrotated version in the positive
            # denominator.  Do all feature preparation outside autocast.
            with torch.autocast(device_type=q.device.type, enabled=False):
                q_den = F.elu(q.float()) + 1.0
                k_den = F.elu(k.float()) + 1.0
                q = rope_apply(q_den, freqs, block.num_heads)
                k = rope_apply(k_den, freqs, block.num_heads)
        else:
            q = rope_apply(q, freqs, block.num_heads)
            k = rope_apply(k, freqs, block.num_heads)
            q_den = q
            k_den = k

        use_gradient_checkpointing = bool(getattr(expert, "use_gradient_checkpointing", False))
        return (
            q,
            k,
            v,
            q_den,
            k_den,
            x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            use_gradient_checkpointing,
        )

    def _apply_post_with_optional_checkpoint(
        self,
        block,
        residual_x: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        use_gradient_checkpointing: bool,
        mixed_slice: torch.Tensor,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        """Apply post-attention computations, with optional checkpointing.

        Args:
            block: Transformer block for current layer.
            residual_x: Residual input tokens before attention update, shape [B, S, D].
            gate_msa: Gating tensor used after mixed self-attention.
            shift_mlp: Shift tensor for MLP input modulation.
            scale_mlp: Scale tensor for MLP input modulation.
            gate_mlp: Gating tensor used after MLP.
            use_gradient_checkpointing: If True and training, checkpoint this post block.
            mixed_slice: Mixed-attention output for this expert, shape [B, S, H*Dh].
            context_payload: Optional dict for cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, S, L] or [B, 1, S, L]

        Returns:
            Updated expert tokens after self-attn residual, optional cross-attn, and MLP.
        """

        def _post_fn(
            _mixed_slice: torch.Tensor,
            _x: torch.Tensor,
            _gate_msa: torch.Tensor,
            _shift_mlp: torch.Tensor,
            _scale_mlp: torch.Tensor,
            _gate_mlp: torch.Tensor,
            _block=block,
            _context_payload=context_payload,
        ) -> torch.Tensor:
            return self._apply_expert_post_block(
                block=_block,
                residual_x=_x,
                mixed_attn_out=_mixed_slice,
                gate_msa=_gate_msa,
                shift_mlp=_shift_mlp,
                scale_mlp=_scale_mlp,
                gate_mlp=_gate_mlp,
                context_payload=_context_payload,
            )

        if use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                _post_fn,
                mixed_slice,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_reentrant=False,
            )
        return _post_fn(
            mixed_slice,
            residual_x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        )

    def prefill_video_cache(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        video_attention_mask: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        """Prefill video branch once and cache per-layer K/V for action denoising.

        Args:
            video_tokens: Video tokens before layer 0, shape [B, Sv, D].
            video_freqs: Video RoPE frequencies, shape [Sv, 1, rope_dim].
            video_t_mod: Video time modulation tensor.
            video_context_payload: Optional dict for video cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, Sv, L] or [B, 1, Sv, L]
            video_attention_mask: Video self-attention mask, shape [Sv, Sv].

        Returns:
            Layer-wise cache list with length `num_layers`.
            Each entry contains:
                - `k`: video key tensor [B, Sv, H*Dh]
                - `v`: video value tensor [B, Sv, H*Dh]
        """
        if "video" not in self.mixtures:
            raise ValueError("MoT requires `video` expert for `prefill_video_cache`.")
        if self.mixed_attention_type == "linear":
            raise NotImplementedError(
                "linear mixed attention does not support the legacy video K/V cache; "
                "use the full joint-sequence forward path"
            )
        if video_attention_mask.ndim != 2:
            raise ValueError(
                f"`video_attention_mask` must be 2D [S,S], got shape {tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape[0] != video_attention_mask.shape[1]:
            raise ValueError(
                f"`video_attention_mask` must be square, got shape {tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape[0] != video_tokens.shape[1]:
            raise ValueError(
                "`video_attention_mask` seq length mismatch: "
                f"mask={video_attention_mask.shape[0]} vs tokens={video_tokens.shape[1]}"
            )

        expert = self.mixtures["video"]
        x = video_tokens
        kv_cache: list[dict[str, torch.Tensor]] = []
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            # Build video Q/K/V from current layer input tokens.
            (
                q,
                k,
                v,
                q_den,
                k_den,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=video_freqs,
                t_mod=video_t_mod,
            )
            # Video prefill uses only video self-attention mask.
            mixed = self._mixed_attention(
                q_cat=q,
                k_cat=k,
                v_cat=v,
                attention_mask=video_attention_mask,
            )
            # Update video tokens for the next layer and persist current layer K/V.
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=video_context_payload,
            )
            kv_cache.append({"k": k, "v": v})
        return kv_cache

    def forward_action_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict],
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor | Sequence[torch.Tensor],
        video_seq_len: int,
    ) -> torch.Tensor:
        """Run action branch with cached video K/V instead of recomputing video tokens.

        Args:
            action_tokens: Action tokens before layer 0, shape [B, Sa, D].
            action_freqs: Action RoPE frequencies, shape [Sa, 1, rope_dim].
            action_t_mod: Action time modulation tensor.
            action_context_payload: Optional dict for action cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, Sa, L] or [B, 1, Sa, L]
            video_kv_cache: Layer-wise cached video K/V from `prefill_video_cache`.
            attention_mask: Joint [video+action] mask, shape [Sv+Sa, Sv+Sa].
            video_seq_len: Video token count `Sv` in the joint sequence prefix.

        Returns:
            Updated action tokens after all layers, shape [B, Sa, D].
        """
        if "action" not in self.mixtures:
            raise ValueError("MoT requires `action` expert for `forward_action_with_video_cache`.")
        if self.mixed_attention_type == "linear":
            raise NotImplementedError(
                "linear mixed attention does not support the legacy video K/V cache; "
                "use the full joint-sequence forward path"
            )
        if len(video_kv_cache) != self.num_layers:
            raise ValueError(
                f"`video_kv_cache` must contain {self.num_layers} layers, got {len(video_kv_cache)}."
            )

        action_seq_len = int(action_tokens.shape[1])
        total_seq_len = int(video_seq_len) + action_seq_len
        attention_masks = self._resolve_attention_masks(
            attention_mask, expected_seq_len=total_seq_len
        )

        expert = self.mixtures["action"]
        x = action_tokens
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            layer_attention_mask = attention_masks[layer_idx]
            action_attention_mask = layer_attention_mask[
                video_seq_len:total_seq_len, :total_seq_len
            ]
            # Action query/key/value are still step-dependent and must be recomputed each step.
            (
                q_action,
                k_action,
                v_action,
                q_action_den,
                k_action_den,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=action_freqs,
                t_mod=action_t_mod,
            )
            layer_cache = video_kv_cache[layer_idx]
            if "k" not in layer_cache or "v" not in layer_cache:
                raise ValueError(f"`video_kv_cache[{layer_idx}]` must contain `k` and `v`.")

            k_video = layer_cache["k"]
            v_video = layer_cache["v"]
            if k_video.shape[1] != video_seq_len or v_video.shape[1] != video_seq_len:
                raise ValueError(
                    f"`video_kv_cache[{layer_idx}]` seq len mismatch, expected {video_seq_len}."
                )

            # Mixed attention: action queries attend to cached video K/V plus current action K/V.
            k_cat = torch.cat([k_video, k_action], dim=1)
            v_cat = torch.cat([v_video, v_action], dim=1)
            mixed = self._mixed_attention(
                q_cat=q_action,
                k_cat=k_cat,
                v_cat=v_cat,
                attention_mask=action_attention_mask,
            )
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=action_context_payload,
            )
        return x

    def forward(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor | Sequence[torch.Tensor],
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
        linear_attention_groups: Optional[Dict[str, torch.Tensor]] = None,
    ):
        missing = [k for k in self.expert_order if k not in embeds_all]
        if missing:
            raise ValueError(f"Missing expert tokens for {missing}")
        missing = [k for k in self.expert_order if k not in freqs_all]
        if missing:
            raise ValueError(f"Missing expert freqs for {missing}")
        missing = [k for k in self.expert_order if k not in t_mod_all]
        if missing:
            raise ValueError(f"Missing expert t_mod for {missing}")

        tokens_all = {k: v for k, v in embeds_all.items()}
        total_seq_len = sum(int(tokens_all[name].shape[1]) for name in self.expert_order)
        attention_masks = self._resolve_attention_masks(
            attention_mask, expected_seq_len=total_seq_len
        )

        for layer_idx in range(self.num_layers):
            q_chunks = []
            k_chunks = []
            v_chunks = []
            q_den_chunks = []
            k_den_chunks = []
            cached = {}
            seq_lens = []

            for name in self.expert_order:
                expert = self.mixtures[name]
                block = expert.blocks[layer_idx]
                x = tokens_all[name]
                freqs = freqs_all[name]
                t_mod = t_mod_all[name]

                (
                    q,
                    k,
                    v,
                    q_den,
                    k_den,
                    residual_x,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                    use_gradient_checkpointing,
                ) = self._build_expert_attention_io(
                    expert=expert,
                    block=block,
                    x=x,
                    freqs=freqs,
                    t_mod=t_mod,
                )

                q_chunks.append(q)
                k_chunks.append(k)
                v_chunks.append(v)
                q_den_chunks.append(q_den)
                k_den_chunks.append(k_den)
                seq_lens.append(x.shape[1])
                cached[name] = {
                    "block": block,
                    "residual_x": residual_x,
                    "gate_msa": gate_msa,
                    "shift_mlp": shift_mlp,
                    "scale_mlp": scale_mlp,
                    "gate_mlp": gate_mlp,
                    "use_gradient_checkpointing": use_gradient_checkpointing,
                }

            # 3. concat all tokens for mixed attention
            q_cat = torch.cat(q_chunks, dim=1)
            k_cat = torch.cat(k_chunks, dim=1)
            v_cat = torch.cat(v_chunks, dim=1)
            q_den_cat = torch.cat(q_den_chunks, dim=1)
            k_den_cat = torch.cat(k_den_chunks, dim=1)

            total_seq = q_cat.shape[1]
            layer_attention_mask = attention_masks[layer_idx]
            if layer_attention_mask.shape[-1] != total_seq:
                raise ValueError(
                    "Attention mask seq length mismatch: "
                    f"mask={layer_attention_mask.shape[-1]} vs tokens={total_seq}"
                )

            mixed = self._mixed_attention(
                q_cat=q_cat,
                k_cat=k_cat,
                v_cat=v_cat,
                attention_mask=layer_attention_mask,
                linear_attention_groups=linear_attention_groups,
                linear_q_den=q_den_cat,
                linear_k_den=k_den_cat,
            )

            start = 0
            for name, seq_len in zip(self.expert_order, seq_lens):
                # 4. split mixed attention output and apply post-attention blocks for each expert
                end = start + seq_len
                mixed_slice = mixed[:, start:end, :]
                cached_expert = cached[name]
                block = cached_expert["block"]
                context_payload = context_all.get(name)

                updated_tokens = self._apply_post_with_optional_checkpoint(
                    block=block,
                    residual_x=cached_expert["residual_x"],
                    gate_msa=cached_expert["gate_msa"],
                    shift_mlp=cached_expert["shift_mlp"],
                    scale_mlp=cached_expert["scale_mlp"],
                    gate_mlp=cached_expert["gate_mlp"],
                    use_gradient_checkpointing=cached_expert["use_gradient_checkpointing"],
                    mixed_slice=mixed_slice,
                    context_payload=context_payload,
                )

                tokens_all[name] = updated_tokens
                start = end

        return tokens_all
