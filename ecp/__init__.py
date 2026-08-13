"""ECP — a post-tool verification runtime for agentic systems.

MCP gets data into your agent. ECP makes sure the final answer only
contains claims your evidence actually supports.

Quick start (zero-config):

    from ecp import VerifiedPipeline
    from ecp.llm import anthropic_llm

    pipeline = VerifiedPipeline(llm=anthropic_llm())
    result = pipeline.run(question, tool_results=tool_results)
    print(result.text)          # verified prose
    print(result.proof)         # machine-readable provenance

Production (curated evidence — recommended):

    from ecp import EvidenceStore, CalcRegistry, VerifiedPipeline

    store = EvidenceStore()
    store.add_value("Q2 unit sales", 100, "units", source_tool="sales_db", ref="SELECT ...")
    ...
    result = pipeline.run(question, store=store)
"""
from .version import __version__
from .store import Evidence, EvidenceStore
from .calc import Calculation, CalcRegistry, CalcError
from .claims import Claim, parse_claims
from .errors import ConfigError, EvidenceOverflowError
from .verifier import Verifier, VerificationResult
from .render import Renderer
from .pipeline import AuditError, LLMError, PipelineConfig, PipelineResult, VerifiedPipeline
from .tier2 import llm_judge_backend
from . import adapters, llm

__all__ = [
    "Evidence", "EvidenceStore", "Calculation", "CalcRegistry", "Claim",
    "parse_claims", "Verifier", "VerificationResult", "Renderer",
    "PipelineConfig", "PipelineResult", "VerifiedPipeline", "CalcError", "LLMError", "AuditError",
    "ConfigError", "EvidenceOverflowError", "llm_judge_backend",
    "adapters", "llm", "__version__",
]
