"""Single source of truth for selected-record verification classification."""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class VerificationPolicy:
    payload_bits: int = 128
    exact_match_ber: float = 0.0
    detection_ber_threshold: float = 0.15
    policy_version: str = "provisional-2026-07"
    calibration_reference: str = "Provisional legacy boundary; labeled calibration pending"

    @property
    def ber_step(self) -> float:
        return 1.0 / self.payload_bits

    def classify(self, ber: Optional[float], error: Optional[str] = None) -> Tuple[str, str]:
        if error:
            return "error", f"Extraction failed: {error}"
        if ber is None:
            return "extraction_failed", "No valid watermark payload could be extracted."
        if ber == self.exact_match_ber:
            return "match", "Exact payload match detected for the selected artwork record."
        if ber <= self.detection_ber_threshold:
            return "partial", (
                "The extracted payload is sufficiently similar to the selected artwork record "
                f"under the provisional policy (BER: {ber:.4f})."
            )
        return "no_match", "No valid watermark match was detected for the selected artwork record."


VERIFICATION_POLICY = VerificationPolicy()
