"""Independent context schedules and disabled-stream invariance."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

_spec = importlib.util.spec_from_file_location('arrival_controls', Path(__file__).resolve().parents[1] / 'protocols/context_arrival/controls.py')
controls = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(controls)

@pytest.mark.parametrize('baseline,expected', [('None',('none','none')), ('Action',('none','arrival')), ('Video',('arrival','none')), ('Full',('arrival','arrival'))])
def test_presets(baseline, expected):
    args=SimpleNamespace(baseline_type=baseline,domain='arrival',video_domain=None,motion_domain=None)
    assert controls.resolve_domains(args)==expected


def test_independent_arrival():
    args=SimpleNamespace(baseline_type='Full',domain='arrival',video_domain='full',motion_domain='arrival')
    v,m=controls.resolve_domains(args)
    assert [(controls.visible_at(v,b),controls.visible_at(m,b)) for b in range(2)]==[(True,False),(True,True)]
