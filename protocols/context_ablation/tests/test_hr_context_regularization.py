"""The sweep must vary full removal without changing partial-mask mass."""

import pytest
import torch

from test_hr_mot import tiny_config
from fasterwam.models.hr_mot import HRMoTFlowModel


@pytest.mark.parametrize("none", [1 / 70, 0.1, 0.3, 0.5])
def test_realized_eight_pattern_probabilities(none):
    full = 1 - 6 / 70 - none
    model = HRMoTFlowModel(tiny_config(full_human_context_prob=full, none_human_context_prob=none))
    masks = model.sample_human_mask(
        300000, device=torch.device("cpu"), generator=torch.Generator().manual_seed(919)
    )
    codes = (masks.long() * torch.tensor([1, 2, 4])).sum(1)
    observed = torch.bincount(codes, minlength=8).float() / len(codes)
    expected = torch.tensor([none, *([1 / 70] * 6), full])
    assert torch.allclose(observed, expected, atol=0.003, rtol=0)


def test_legacy_sampler_and_rng_are_unchanged():
    model = HRMoTFlowModel(tiny_config())
    rng = torch.Generator().manual_seed(77)
    actual = model.sample_human_mask(2048, device=torch.device("cpu"), generator=rng)
    expected_rng = torch.Generator().manual_seed(77)
    choices = torch.randint(0, 7, (2048,), generator=expected_rng)
    expected = ((choices[:, None] >> torch.arange(3)) & 1).bool()
    full = torch.rand(2048, generator=expected_rng) < 0.9
    expected[full] = True
    assert torch.equal(actual, expected)
    assert torch.equal(rng.get_state(), expected_rng.get_state())


@pytest.mark.parametrize(
    "none,full,pin",
    [(-0.1, 0.9, False), (0.3, 0.9, False), (0.3, 0.5, True), (float("nan"), 0.5, False)],
)
def test_invalid_probability_contract_fails(none, full, pin):
    with pytest.raises(ValueError):
        tiny_config(
            none_human_context_prob=none,
            full_human_context_prob=full,
            droppable_human_current=not pin,
        ).validate()
