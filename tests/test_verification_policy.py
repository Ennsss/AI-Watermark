from pathlib import Path
import sys

BACKEND_DIR = Path(__file__).resolve().parents[1] / "web-app" / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from verification_policy import VERIFICATION_POLICY


def test_policy_boundaries_use_representable_ber_steps():
    policy = VERIFICATION_POLICY
    step = policy.ber_step
    boundary_bits = int(policy.detection_ber_threshold * policy.payload_bits)
    representable_boundary = boundary_bits / policy.payload_bits

    assert policy.classify(0)[0] == "match"
    assert policy.classify(representable_boundary)[0] == "partial"
    assert policy.classify(representable_boundary - step)[0] == "partial"
    assert policy.classify(representable_boundary + step)[0] == "no_match"
    assert policy.classify(None)[0] == "extraction_failed"
    assert policy.classify(None, "decode error")[0] == "error"


def test_no_match_message_preserves_selected_record_scope():
    status, message = VERIFICATION_POLICY.classify(0.5)
    assert status == "no_match"
    assert "selected artwork record" in message.lower()
    assert "never watermarked" not in message.lower()


def test_committed_policy_is_explicitly_provisional():
    assert VERIFICATION_POLICY.policy_version.startswith("provisional")
    assert "pending" in VERIFICATION_POLICY.calibration_reference.lower()
