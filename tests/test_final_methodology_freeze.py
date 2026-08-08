"""Non-test-data validation of the guarded final benchmark."""
import json
from pathlib import Path
import numpy as np

from scripts import final_heldout_benchmark as final


def test_frozen_payload_is_balanced_and_reproducible():
    cfg=final.load_config()
    bits=final.payload_from_config(cfg)
    assert bits.shape==(128,)
    assert np.sum(bits==0)==64
    assert np.sum(bits==1)==64


def test_checkpoint_hash_verifies_without_model_execution():
    path,digest=final.verify_checkpoint(final.load_config())
    assert path.name=="best_seed_aware_delta24.keras"
    assert digest=="afae879e3078543d3be030889b277a90245c3c73b041b912a209b1489729e53d"


def test_default_guard_refuses_before_final_run(monkeypatch):
    called=False
    def forbidden(_cfg):
        nonlocal called
        called=True
    monkeypatch.setattr(final,"run_final",forbidden)
    assert final.main([])==2
    assert not called


def test_validate_freeze_does_not_start_final_run(monkeypatch):
    monkeypatch.setattr(final,"run_final",lambda _cfg: (_ for _ in ()).throw(AssertionError("must not run")))
    assert final.main(["--validate-freeze"])==0


def test_crop_dispatch_is_deterministic_on_synthetic_array():
    cfg=final.load_config();image=np.arange(512*512*3,dtype=np.uint32).reshape(512,512,3).astype(np.uint8)
    first=final.apply_condition(image,"crop_mild",cfg)
    second=final.apply_condition(image,"crop_mild",cfg)
    assert np.array_equal(first,second)
    assert first.shape==(494,512,3)


def test_statistics_helpers_use_positive_cnn_advantage():
    classical=np.array([.5,.4,.3,.2]);cnn=np.array([.4,.3,.3,.1])
    assert final.rank_biserial(classical,cnn)>0
    adjusted=final.holm_adjust([.01,.04,.2])
    assert np.allclose(adjusted,[.03,.08,.2])


def test_results_directory_contains_no_numeric_placeholders():
    directory=Path("experiments/final_heldout_benchmark")
    assert sorted(p.name for p in directory.iterdir())==["README.md"]
    assert not any((directory/name).exists() for name in final.EXPECTED_FILES)
