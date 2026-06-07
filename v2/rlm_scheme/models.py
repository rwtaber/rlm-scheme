"""Pydantic v2 models for all durable records, API payloads, and config."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TaskShape(str, enum.Enum):
    Direct = "Direct"
    Batch = "Batch"
    Synthesize = "Synthesize"
    Search = "Search"
    Refine = "Refine"
    Compare = "Compare"
    Classify = "Classify"
    Pipeline = "Pipeline"
    Generate = "Generate"
    Decompose = "Decompose"
    Validate = "Validate"
    Aggregate = "Aggregate"
    Composite = "Composite"


class DataShape(str, enum.Enum):
    FlatList = "FlatList"
    Hierarchy = "Hierarchy"
    Singular = "Singular"
    ChunkedSingular = "ChunkedSingular"
    Graph = "Graph"
    TimeSeries = "TimeSeries"
    Tabular = "Tabular"
    Multimodal = "Multimodal"
    Paired = "Paired"
    KeyValue = "KeyValue"
    Unknown = "Unknown"


class ExecutionState(str, enum.Enum):
    queued = "queued"
    running = "running"
    finished = "finished"
    failed = "failed"
    cancelled = "cancelled"
    awaiting_gate = "awaiting_gate"
    gate_rejected = "gate_rejected"


class VerificationDecision(str, enum.Enum):
    pass_ = "pass"
    warn = "warn"
    fail = "fail"


class ErrorPolicy(str, enum.Enum):
    fail_fast = "fail_fast"
    collect = "collect"
    fallback = "fallback"


class ResetScope(str, enum.Enum):
    sandbox = "sandbox"
    session = "session"
    all = "all"
    cache = "cache"
    config = "config"


# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------

class DataRef(BaseModel):
    storage: str = "filesystem"
    path: str
    hash: str
    bytes: int


class ChunkingInfo(BaseModel):
    chunk_count: int | None = None
    overlap_tokens: int | None = None
    boundary: str | None = None


class SourceInfo(BaseModel):
    kind: str = "inline"
    uri: str | None = None


class ContextMetadata(BaseModel):
    data_shape: DataShape | None = None
    item_count: int | None = None
    item_size_estimate_tokens: int | None = None
    total_size_estimate_tokens: int | None = None
    independent: bool | None = None
    ordered: bool | None = None
    modality: list[str] | None = None
    chunking: ChunkingInfo | None = None
    source: SourceInfo | None = None
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Durable record models (section 5)
# ---------------------------------------------------------------------------

class ContextRecord(BaseModel):
    context_id: str
    schema_version: str = "1"
    name: str | None = None
    created_at: datetime
    data_ref: DataRef
    metadata: ContextMetadata


class Classification(BaseModel):
    task_shape: TaskShape
    constituent_shapes: list[TaskShape] | None = None
    data_shape: DataShape
    confidence: float
    rationale: str | None = None


class TemplateInvocation(BaseModel):
    template_name: str
    template_version: str
    slot_values: dict[str, Any] = Field(default_factory=dict)


class TemplateChainStep(BaseModel):
    template_name: str
    template_version: str
    slot_values: dict[str, Any] = Field(default_factory=dict)


class RecommendedAction(BaseModel):
    kind: str  # "template_invocation" | "template_chain" | "no_template"
    template_name: str | None = None
    template_version: str | None = None
    slot_values: dict[str, Any] | None = None
    steps: list[TemplateChainStep] | None = None
    # no_template fields
    description: str | None = None
    needed_template: dict[str, Any] | None = None


class PlannerInfo(BaseModel):
    mode: str = "deterministic_with_llm_fill"
    model: str | None = None
    prompt_hash: str | None = None


class AlternativeTemplate(BaseModel):
    template_name: str
    tradeoff: str


class PlanRecord(BaseModel):
    plan_id: str
    schema_version: str = "1"
    created_at: datetime
    context_ids: list[str] = Field(default_factory=list)
    task: str
    hints: dict[str, Any] = Field(default_factory=dict)
    classification: Classification
    recommended: RecommendedAction
    alternatives: list[AlternativeTemplate] = Field(default_factory=list)
    planner: PlannerInfo = Field(default_factory=PlannerInfo)


class StaticProfile(BaseModel):
    expected_calls_formula: str | None = None
    max_concurrency: int | None = None
    recursive_depth: int = 0
    uses_python_bridge: bool = False
    uses_multimodal: bool = False


class InstantiatorInfo(BaseModel):
    name: str = "rlm-scheme-template-instantiator"
    version: str = "0.1.0"


class ArtifactRecord(BaseModel):
    artifact_id: str
    schema_version: str = "1"
    created_at: datetime
    plan_id: str | None = None
    context_ids: list[str] = Field(default_factory=list)
    source_type: str = "template_invocation"
    template_name: str
    template_version: str
    slot_values: dict[str, Any] = Field(default_factory=dict)
    instantiator: InstantiatorInfo = Field(default_factory=InstantiatorInfo)
    generated_scheme_ref: DataRef
    primitives_used: list[str] = Field(default_factory=list)
    static_profile: StaticProfile = Field(default_factory=StaticProfile)


class CallGraphNode(BaseModel):
    node_id: str
    primitive: str
    calls: int
    model: str | None = None
    concurrency: int | None = None
    branch_factor: int | None = None


class DryRunSummary(BaseModel):
    llm_calls: int = 0
    max_concurrency: int = 0
    recursive_depth: int = 0
    critical_path_calls: int = 0
    checkpoints: int = 0
    python_phases: int = 0


class DryRunRecord(BaseModel):
    dry_run_id: str
    schema_version: str = "1"
    created_at: datetime
    artifact_id: str
    plan_id: str | None = None
    mode: str = "deterministic"
    summary: DryRunSummary = Field(default_factory=DryRunSummary)
    call_graph: list[CallGraphNode] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class VerificationCheck(BaseModel):
    name: str
    status: str  # "pass" | "warn" | "fail"
    message: str


class VerificationRecord(BaseModel):
    verification_id: str
    schema_version: str = "1"
    created_at: datetime
    artifact_id: str
    dry_run_id: str | None = None
    decision: VerificationDecision
    policy: dict[str, Any] = Field(default_factory=dict)
    checks: list[VerificationCheck] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class GateRecord(BaseModel):
    name: str
    status: str = "pending"  # pending | approved | rejected
    decided_at: datetime | None = None
    reason: str | None = None


class ChainStepResult(BaseModel):
    step: int
    template: str
    intermediate_context_id: str | None = None
    result: Any = None


class ExecutionMetrics(BaseModel):
    elapsed_seconds: float = 0.0
    llm_calls: int = 0
    tokens: int = 0
    max_concurrency_observed: int = 0
    cache_hits: int = 0
    budget_policy_activations: int = 0


class ExecutionRecord(BaseModel):
    execution_id: str
    schema_version: str = "1"
    created_at: datetime
    completed_at: datetime | None = None
    state: ExecutionState = ExecutionState.queued
    artifact_id: str
    plan_id: str | None = None
    verification_id: str | None = None
    result_ref: DataRef | None = None
    trace_ref: DataRef | None = None
    metrics: ExecutionMetrics = Field(default_factory=ExecutionMetrics)
    error: dict[str, Any] | None = None
    gates: list[GateRecord] = Field(default_factory=list)
    chain_step_results: list[ChainStepResult] | None = None


class TokenUsage(BaseModel):
    prompt: int = 0
    completion: int = 0
    total: int = 0


class CacheRecord(BaseModel):
    cache_key: str
    schema_version: str = "1"
    created_at: datetime
    instruction_hash: str
    data_hash: str
    model: str
    temperature: float
    json_mode: bool = False
    result: str
    result_tokens: TokenUsage = Field(default_factory=TokenUsage)
    source_execution_id: str | None = None
    source_call_id: str | None = None


# ---------------------------------------------------------------------------
# API response models (section 4)
# ---------------------------------------------------------------------------

class MCPResponse(BaseModel):
    status: str = "ok"
    id: str | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)


class LoadContextResponse(MCPResponse):
    context_id: str
    name: str | None = None
    metadata: ContextMetadata | None = None
    preview: str | None = None


class ContextDetail(BaseModel):
    context_id: str
    name: str | None = None
    created_at: datetime
    metadata: ContextMetadata | None = None
    preview: str | None = None
    data_hash: str | None = None
    data: Any = None
    data_too_large: bool | None = None


class GetContextResponse(MCPResponse):
    context: ContextDetail | None = None


class PlanStrategyResponse(MCPResponse):
    plan_id: str
    classification: Classification
    recommended: RecommendedAction
    alternatives: list[AlternativeTemplate] = Field(default_factory=list)


class ArtifactSummary(BaseModel):
    artifact_id: str
    template_name: str
    template_version: str
    code_hash: str
    primitives_used: list[str] = Field(default_factory=list)


class CostEstimate(BaseModel):
    low: float
    high: float


class TokenEstimate(BaseModel):
    prompt: int
    completion: int
    total: int


class DryRunEstimate(BaseModel):
    expected_llm_calls: int
    critical_path_calls: int
    max_concurrency: int
    models: dict[str, int] = Field(default_factory=dict)
    estimated_tokens: TokenEstimate | None = None
    estimated_cost_usd: CostEstimate | None = None


class DryRunSimulation(BaseModel):
    llm_calls: int = 0
    max_concurrency: int = 0
    recursive_depth: int = 0
    critical_path_calls: int = 0
    checkpoints: int = 0
    python_phases: int = 0


class DryRunResponse(MCPResponse):
    dry_run_id: str
    plan_id: str | None = None
    artifact: ArtifactSummary | None = None
    estimate: DryRunEstimate | None = None
    simulation: DryRunSimulation | None = None
    call_graph: list[CallGraphNode] = Field(default_factory=list)
    output_schema: Any = None
    cache_hits_expected: int | None = None


class VerificationSummary(BaseModel):
    verification_id: str
    decision: str
    checks: list[VerificationCheck] = Field(default_factory=list)


class ExecutionResult(BaseModel):
    value: Any = None
    stdout: str | None = None


class ExecutionSummary(BaseModel):
    state: str
    elapsed_seconds: float = 0.0
    llm_calls: int = 0
    tokens: int = 0
    models: dict[str, int] = Field(default_factory=dict)
    checkpoints_written: int = 0
    cache_hits: int = 0
    budget_policy_activations: int = 0
    chain_step_results: list[ChainStepResult] | None = None
    gate: dict[str, Any] | None = None


class ExecuteResponse(MCPResponse):
    execution_id: str | None = None
    artifact_id: str | None = None
    verification: VerificationSummary | None = None
    result: ExecutionResult | None = None
    execution: ExecutionSummary | None = None


class TraceEvent(BaseModel):
    type: str
    call_id: str | None = None
    node_id: str | None = None
    model: str | None = None
    depth: int | None = None
    tokens: int | None = None
    elapsed_seconds: float | None = None


class ScopeLogEntry(BaseModel):
    op: str
    preview: str | None = None
    scope: str | None = None
    call_id: str | None = None


class TraceDetail(BaseModel):
    artifact_id: str | None = None
    plan_id: str | None = None
    events: list[TraceEvent] = Field(default_factory=list)
    scope_log: list[ScopeLogEntry] = Field(default_factory=list)
    stdout: str = ""


class TraceResponse(MCPResponse):
    execution_id: str | None = None
    trace: TraceDetail | None = None


class ActiveCall(BaseModel):
    call_id: str
    execution_id: str | None = None
    type: str | None = None
    model: str | None = None
    elapsed_seconds: float | None = None
    depth: int | None = None
    instruction_preview: str | None = None


class RuntimeStatus(BaseModel):
    racket_alive: bool = False
    python_bridge_alive: bool = False
    sandbox_memory_limit_mb: int = 256


class TokenUsageStatus(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0


class RateLimitStatus(BaseModel):
    remaining_requests: int | None = None
    remaining_tokens: int | None = None
    reset_requests: str | None = None


class StatusResponse(MCPResponse):
    runtime: RuntimeStatus = Field(default_factory=RuntimeStatus)
    active_calls: list[ActiveCall] = Field(default_factory=list)
    token_usage: TokenUsageStatus = Field(default_factory=TokenUsageStatus)
    rate_limits: RateLimitStatus = Field(default_factory=RateLimitStatus)


class CancelResponse(MCPResponse):
    cancelled: dict[str, Any] | None = None


class ResumeResponse(MCPResponse):
    execution_id: str | None = None
    gate: dict[str, Any] | None = None
    result: ExecutionResult | None = None
    execution: ExecutionSummary | None = None


class ResetResponse(MCPResponse):
    scope: str | None = None
    cleared: dict[str, int] | None = None


# ---------------------------------------------------------------------------
# Config models (sections 8, 11.6)
# ---------------------------------------------------------------------------

class ModelAlias(BaseModel):
    provider: str
    model: str
    capabilities: list[str] = Field(default_factory=list)
    max_context_tokens: int = 128000
    supports_temperature: bool = True
    cost_tier: str = "low"
    fallback: str | None = None


class ModelRegistry(BaseModel):
    schema_version: str = "1"
    aliases: dict[str, ModelAlias] = Field(default_factory=dict)
    defaults: dict[str, str] = Field(default_factory=dict)


class RetryConfig(BaseModel):
    max_retries: int = 3
    initial_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 60.0
    backoff_multiplier: float = 2.0
    retryable_status_codes: list[int] = Field(
        default_factory=lambda: [429, 500, 502, 503, 504]
    )
    retryable_error_types: list[str] = Field(
        default_factory=lambda: ["rate_limit", "timeout", "server_error"]
    )


class RetrySettings(BaseModel):
    schema_version: str = "1"
    defaults: RetryConfig = Field(default_factory=RetryConfig)
    per_model_overrides: dict[str, RetryConfig] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Execution policy
# ---------------------------------------------------------------------------

class ExecutionPolicy(BaseModel):
    max_llm_calls: int = 1000
    max_concurrency: int = 50
    max_recursive_depth: int = 3
    allow_python_bridge: bool = True
    allow_multimodal: bool = True
    allow_llm_generated_code: bool = False
    require_gates: bool = False
    require_output_schema: bool = False
