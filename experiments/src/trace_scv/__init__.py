"""Controlled multimodal evidence-selection research utilities."""

from .cefes import CEFESDecision, select_cefes_prefix
from .schema import ClaimPacket, EvidenceItem, Prediction

__all__ = [
    "CEFESDecision",
    "ClaimPacket",
    "EvidenceItem",
    "Prediction",
    "select_cefes_prefix",
]
__version__ = "0.4.1"
