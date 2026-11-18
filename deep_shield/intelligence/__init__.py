"""Cross-module evidence linking and integrated response schemas."""

from .linker import link_module_results
from .schema import FindingGroup, FindingLink, IntelligenceResult, LinkRequest

__all__ = [
    "FindingGroup",
    "FindingLink",
    "IntelligenceResult",
    "LinkRequest",
    "link_module_results",
]
