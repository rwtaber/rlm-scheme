"""Tests for Pydantic model schemas — round-trip serialization, defaults, enums."""

import json
from datetime import datetime, timezone

import pytest

from rlm_scheme.models import (
    ArtifactRecord,
    CacheRecord,
    Classification,
    ContextMetadata,
    ContextRecord,
    CostEstimate,
    DataRef,
    DataShape,
    DryRunEstimate,
    DryRunRecord,
    DryRunResponse,
    DryRunSummary,
    ErrorPolicy,
    ExecuteResponse,
    ExecutionMetrics,
    ExecutionPolicy,
    ExecutionRecord,
    ExecutionState,
    GateRecord,
    GetContextResponse,
    LoadContextResponse,
    MCPResponse,
    ModelAlias,
    ModelRegistry,
    PlanRecord,
    PlanStrategyResponse,
    RecommendedAction,
    ResetResponse,
    ResetScope,
    RetryConfig,
    RetrySettings,
    StatusResponse,
    TaskShape,
    TemplateInvocation,
    TokenUsage,
    TraceResponse,
    VerificationCheck,
    VerificationDecision,
    VerificationRecord,
)


NOW = datetime.now(timezone.utc)


def make_data_ref(**overrides):
    defaults = {"storage": "filesystem", "path": "/tmp/test", "hash": "abc123", "bytes": 100}
    defaults.update(overrides)
    return DataRef(**defaults)


class TestEnums:
    def test_task_shapes(self):
        assert len(TaskShape) == 13
        assert TaskShape.Direct.value == "Direct"
        assert TaskShape.Composite.value == "Composite"

    def test_data_shapes(self):
        assert len(DataShape) == 11
        assert DataShape.FlatList.value == "FlatList"

    def test_execution_states(self):
        assert len(ExecutionState) == 7
        assert ExecutionState.awaiting_gate.value == "awaiting_gate"

    def test_verification_decision(self):
        assert VerificationDecision.pass_.value == "pass"
        assert VerificationDecision.warn.value == "warn"
        assert VerificationDecision.fail.value == "fail"

    def test_error_policy(self):
        assert ErrorPolicy.fail_fast.value == "fail_fast"
        assert ErrorPolicy.collect.value == "collect"
        assert ErrorPolicy.fallback.value == "fallback"

    def test_reset_scope(self):
        assert len(ResetScope) == 5


class TestContextRecord:
    def test_round_trip(self):
        rec = ContextRecord(
            context_id="ctx_0123456789abcdef",
            created_at=NOW,
            data_ref=make_data_ref(),
            metadata=ContextMetadata(data_shape=DataShape.FlatList, item_count=10),
        )
        data = rec.model_dump()
        restored = ContextRecord.model_validate(data)
        assert restored.context_id == rec.context_id
        assert restored.metadata.data_shape == DataShape.FlatList
        assert restored.metadata.item_count == 10

    def test_json_serialization(self):
        rec = ContextRecord(
            context_id="ctx_0123456789abcdef",
            created_at=NOW,
            data_ref=make_data_ref(),
            metadata=ContextMetadata(),
        )
        json_str = rec.model_dump_json()
        restored = ContextRecord.model_validate_json(json_str)
        assert restored.context_id == rec.context_id


class TestPlanRecord:
    def test_minimal(self):
        rec = PlanRecord(
            plan_id="plan_0123456789abcdef",
            created_at=NOW,
            task="Extract data",
            classification=Classification(
                task_shape=TaskShape.Batch,
                data_shape=DataShape.FlatList,
                confidence=0.95,
            ),
            recommended=RecommendedAction(
                kind="template_invocation",
                template_name="batch_map",
                template_version="1.0.0",
                slot_values={"max_concurrent": 20},
            ),
        )
        data = rec.model_dump()
        assert data["classification"]["task_shape"] == "Batch"
        restored = PlanRecord.model_validate(data)
        assert restored.recommended.template_name == "batch_map"


class TestArtifactRecord:
    def test_defaults(self):
        rec = ArtifactRecord(
            artifact_id="art_0123456789abcdef",
            created_at=NOW,
            template_name="direct_call",
            template_version="1.0.0",
            generated_scheme_ref=make_data_ref(),
        )
        assert rec.source_type == "template_invocation"
        assert rec.instantiator.name == "rlm-scheme-template-instantiator"
        assert rec.static_profile.recursive_depth == 0


class TestDryRunRecord:
    def test_defaults(self):
        rec = DryRunRecord(
            dry_run_id="dry_0123456789abcdef",
            created_at=NOW,
            artifact_id="art_0123456789abcdef",
        )
        assert rec.summary.llm_calls == 0
        assert rec.mode == "deterministic"


class TestVerificationRecord:
    def test_with_checks(self):
        rec = VerificationRecord(
            verification_id="ver_0123456789abcdef",
            created_at=NOW,
            artifact_id="art_0123456789abcdef",
            decision=VerificationDecision.pass_,
            checks=[
                VerificationCheck(name="hash_match", status="pass", message="OK"),
                VerificationCheck(name="primitives_only", status="pass", message="OK"),
            ],
        )
        assert len(rec.checks) == 2
        assert rec.decision == VerificationDecision.pass_


class TestExecutionRecord:
    def test_state_transitions(self):
        rec = ExecutionRecord(
            execution_id="exec_0123456789abcdef",
            created_at=NOW,
            artifact_id="art_0123456789abcdef",
        )
        assert rec.state == ExecutionState.queued
        rec.state = ExecutionState.running
        assert rec.state == ExecutionState.running
        rec.state = ExecutionState.finished
        assert rec.state == ExecutionState.finished

    def test_gate_records(self):
        rec = ExecutionRecord(
            execution_id="exec_0123456789abcdef",
            created_at=NOW,
            artifact_id="art_0123456789abcdef",
            gates=[GateRecord(name="review", status="pending")],
        )
        assert rec.gates[0].name == "review"


class TestCacheRecord:
    def test_round_trip(self):
        rec = CacheRecord(
            cache_key="cache_0123456789abcdef",
            created_at=NOW,
            instruction_hash="sha256_abc",
            data_hash="sha256_def",
            model="fast_text_model",
            temperature=0.0,
            result="cached result",
            result_tokens=TokenUsage(prompt=100, completion=50, total=150),
        )
        data = rec.model_dump()
        restored = CacheRecord.model_validate(data)
        assert restored.result == "cached result"
        assert restored.result_tokens.total == 150


class TestAPIResponses:
    def test_mcp_response_defaults(self):
        resp = MCPResponse()
        assert resp.status == "ok"
        assert resp.warnings == []
        assert resp.errors == []

    def test_load_context_response(self):
        resp = LoadContextResponse(
            context_id="ctx_0123456789abcdef",
            name="test",
            preview="first 200 chars...",
        )
        assert resp.status == "ok"
        assert resp.context_id == "ctx_0123456789abcdef"

    def test_plan_strategy_response(self):
        resp = PlanStrategyResponse(
            plan_id="plan_0123456789abcdef",
            classification=Classification(
                task_shape=TaskShape.Direct,
                data_shape=DataShape.Singular,
                confidence=1.0,
            ),
            recommended=RecommendedAction(kind="template_invocation"),
        )
        assert resp.plan_id.startswith("plan_")

    def test_dry_run_response(self):
        resp = DryRunResponse(dry_run_id="dry_0123456789abcdef")
        assert resp.dry_run_id.startswith("dry_")

    def test_execute_response(self):
        resp = ExecuteResponse(execution_id="exec_0123456789abcdef")
        assert resp.status == "ok"

    def test_status_response_defaults(self):
        resp = StatusResponse()
        assert resp.runtime.racket_alive is False
        assert resp.runtime.sandbox_memory_limit_mb == 256

    def test_reset_response(self):
        resp = ResetResponse(scope="session", cleared={"contexts": 5, "plans": 3})
        assert resp.cleared["contexts"] == 5


class TestConfigModels:
    def test_model_alias(self):
        alias = ModelAlias(
            provider="openai",
            model="gpt-4o",
            capabilities=["text", "json", "image"],
            cost_tier="high",
        )
        assert "image" in alias.capabilities
        assert alias.fallback is None

    def test_model_registry_from_json(self):
        raw = {
            "schema_version": "1",
            "aliases": {
                "fast_text_model": {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "capabilities": ["text", "json"],
                    "max_context_tokens": 128000,
                    "supports_temperature": True,
                    "cost_tier": "low",
                }
            },
            "defaults": {"planner": "fast_text_model"},
        }
        reg = ModelRegistry.model_validate(raw)
        assert "fast_text_model" in reg.aliases
        assert reg.defaults["planner"] == "fast_text_model"

    def test_retry_config_defaults(self):
        cfg = RetryConfig()
        assert cfg.max_retries == 3
        assert cfg.backoff_multiplier == 2.0
        assert 429 in cfg.retryable_status_codes

    def test_retry_settings(self):
        settings = RetrySettings(
            per_model_overrides={
                "fast_text_model": RetryConfig(max_retries=5, initial_backoff_seconds=0.5),
            }
        )
        assert settings.per_model_overrides["fast_text_model"].max_retries == 5
        assert settings.defaults.max_retries == 3


class TestExecutionPolicy:
    def test_defaults(self):
        policy = ExecutionPolicy()
        assert policy.max_llm_calls == 1000
        assert policy.max_concurrency == 50
        assert policy.max_recursive_depth == 3
        assert policy.allow_python_bridge is True
        assert policy.allow_llm_generated_code is False

    def test_override(self):
        policy = ExecutionPolicy(max_llm_calls=10, require_gates=True)
        assert policy.max_llm_calls == 10
        assert policy.require_gates is True
