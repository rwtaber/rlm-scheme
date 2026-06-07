"""Executor: instantiate + verify + execute via PythonRuntimeStub."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from typing import Any

from .cache import LLMCache
from .context_store import ContextStore
from .gate import GateManager
from .ids import EXEC_PREFIX, VER_PREFIX, generate_id
from .instantiator import InstantiationError, get_instantiated_code, instantiate
from .llm_provider import LLMProvider
from .models import (
    DataRef,
    ExecuteResponse,
    ExecutionMetrics,
    ExecutionPolicy,
    ExecutionRecord,
    ExecutionResult,
    ExecutionState,
    ExecutionSummary,
    VerificationCheck,
    VerificationDecision,
    VerificationRecord,
    VerificationSummary,
)
from .store import Store
from .template_store import Template, TemplateStore
from .trace import TraceCollector


class Executor:
    """Executes instantiated templates via the PythonRuntimeStub."""

    def __init__(
        self,
        store: Store,
        context_store: ContextStore,
        template_store: TemplateStore,
        llm_provider: LLMProvider,
        cache: LLMCache | None = None,
        gate_manager: GateManager | None = None,
    ) -> None:
        self.store = store
        self.context_store = context_store
        self.template_store = template_store
        self.llm_provider = llm_provider
        self.cache = cache
        self.gate_manager = gate_manager or GateManager()

    async def execute(
        self,
        template: Template,
        slot_values: dict[str, Any],
        context_ids: list[str] | None = None,
        plan_id: str | None = None,
        policy: ExecutionPolicy | None = None,
        timeout: float | None = None,
        stream: bool = False,
    ) -> ExecuteResponse:
        """Full execution pipeline: instantiate -> verify -> execute."""
        policy = policy or ExecutionPolicy()
        context_ids = context_ids or []

        # Step 1: Instantiate
        try:
            artifact = instantiate(
                template, slot_values,
                context_ids=context_ids,
                plan_id=plan_id,
                store=self.store,
            )
        except InstantiationError as e:
            return ExecuteResponse(
                status="error",
                errors=[str(e)],
            )

        # Step 2: Verify
        code = get_instantiated_code(artifact.artifact_id, self.store)
        verification = _verify(artifact, code or "", template, policy, self.context_store, context_ids)
        ver_id = generate_id(VER_PREFIX)
        ver_record = VerificationRecord(
            verification_id=ver_id,
            created_at=datetime.now(timezone.utc),
            artifact_id=artifact.artifact_id,
            decision=verification.decision,
            checks=verification.checks,
            warnings=[c.message for c in verification.checks if c.status == "warn"],
            errors=[c.message for c in verification.checks if c.status == "fail"],
        )
        self.store.save("verifications", ver_id, ver_record.model_dump(mode="json"))

        if verification.decision == VerificationDecision.fail:
            return ExecuteResponse(
                status="error",
                artifact_id=artifact.artifact_id,
                verification=VerificationSummary(
                    verification_id=ver_id,
                    decision="fail",
                    checks=verification.checks,
                ),
                errors=[c.message for c in verification.checks if c.status == "fail"],
            )

        # Step 3: Execute
        exec_id = generate_id(EXEC_PREFIX)
        exec_record = ExecutionRecord(
            execution_id=exec_id,
            created_at=datetime.now(timezone.utc),
            artifact_id=artifact.artifact_id,
            plan_id=plan_id,
            verification_id=ver_id,
            state=ExecutionState.running,
        )
        self.store.save("executions", exec_id, exec_record.model_dump(mode="json"))

        trace = TraceCollector(exec_id, artifact.artifact_id, plan_id)
        start = time.monotonic()

        try:
            result_value = await asyncio.wait_for(
                self._run_stub(code or "", template, slot_values, context_ids, policy, exec_id, trace),
                timeout=timeout,
            )
            elapsed = time.monotonic() - start

            exec_record.state = ExecutionState.finished
            exec_record.completed_at = datetime.now(timezone.utc)
            exec_record.metrics = ExecutionMetrics(
                elapsed_seconds=elapsed,
                llm_calls=self.llm_provider.total_calls,
                tokens=self.llm_provider.total_tokens.total,
                cache_hits=self.cache.hits if self.cache else 0,
            )

            # Persist result
            result_bytes = json.dumps(result_value, default=str).encode()
            result_path = self.store.save_raw("executions", exec_id, "result.json", result_bytes)
            exec_record.result_ref = DataRef(
                storage="filesystem",
                path=str(result_path),
                hash=hashlib.sha256(result_bytes).hexdigest(),
                bytes=len(result_bytes),
            )

        except asyncio.TimeoutError:
            elapsed = time.monotonic() - start
            exec_record.state = ExecutionState.failed
            exec_record.error = {"type": "timeout", "message": f"Execution timed out after {timeout}s"}
            result_value = None

        except Exception as e:
            elapsed = time.monotonic() - start
            exec_record.state = ExecutionState.failed
            exec_record.error = {"type": type(e).__name__, "message": str(e)}
            result_value = None

        # Persist trace and execution record
        trace.persist(self.store)
        self.store.save("executions", exec_id, exec_record.model_dump(mode="json"))

        return ExecuteResponse(
            execution_id=exec_id,
            artifact_id=artifact.artifact_id,
            verification=VerificationSummary(
                verification_id=ver_id,
                decision=verification.decision.value,
                checks=verification.checks,
            ),
            result=ExecutionResult(value=result_value) if result_value is not None else None,
            execution=ExecutionSummary(
                state=exec_record.state.value,
                elapsed_seconds=elapsed,
                llm_calls=self.llm_provider.total_calls,
                tokens=self.llm_provider.total_tokens.total,
                cache_hits=self.cache.hits if self.cache else 0,
            ),
            warnings=[c.message for c in verification.checks if c.status == "warn"],
        )

    async def _run_stub(
        self,
        code: str,
        template: Template,
        slot_values: dict[str, Any],
        context_ids: list[str],
        policy: ExecutionPolicy,
        exec_id: str,
        trace: TraceCollector,
    ) -> Any:
        """PythonRuntimeStub: pattern-match on known template structures."""
        # Detect which primitives are present
        has_map = "map-async" in code
        has_tree_reduce = "tree-reduce" in code
        has_fold = "fold-sequential" in code
        has_iterate = "iterate-until" in code
        has_parallel = "parallel" in code
        has_race = "race" in code

        # Get context data
        ctx_id = slot_values.get("context_id")
        items_path = slot_values.get("items_path", "$")

        if ctx_id and context_ids:
            try:
                data = self.context_store.extract_path(ctx_id, items_path)
            except Exception:
                data = self.context_store.get_data(ctx_id) if ctx_id else None
        else:
            data = None

        items = data if isinstance(data, list) else [data] if data is not None else ["[stub data]"]
        max_concurrent = slot_values.get("max_concurrent", 20)

        # Map phase
        map_results: list[str] = []
        if has_map:
            instruction = slot_values.get("map_instruction", slot_values.get("instruction", "process"))
            model = slot_values.get("map_model", slot_values.get("model", "fast_text_model"))
            json_mode = slot_values.get("json_mode", False)

            sem = asyncio.Semaphore(max_concurrent)
            async def process_item(item: Any) -> str:
                async with sem:
                    # Check cache
                    if self.cache:
                        cached = self.cache.lookup(instruction, item, model, 0.0)
                        if cached:
                            trace.record_cache_hit(f"call_{len(map_results)}")
                            return cached.result

                    call_id = f"call_{len(trace.events):04d}"
                    trace.record_call_start(call_id, model)
                    result = await self.llm_provider.call(
                        instruction=instruction, data=item,
                        model=model, json_mode=json_mode,
                    )
                    trace.record_call_end(call_id, tokens=result.tokens.total, model=model)

                    if self.cache:
                        self.cache.store_result(
                            instruction, item, model, 0.0,
                            result.text, result.tokens, json_mode,
                            execution_id=exec_id,
                        )
                    return result.text

            map_results = await asyncio.gather(*[process_item(item) for item in items])

        # Reduce phase
        if has_tree_reduce and map_results:
            instruction = slot_values.get("reduce_instruction", "synthesize")
            model = slot_values.get("reduce_model", "quality_text_model")
            branch_factor = slot_values.get("branch_factor", 5)
            result = await self._tree_reduce(map_results, instruction, model, branch_factor, trace)
            return result

        if has_fold and map_results:
            instruction = slot_values.get("fold_instruction", "synthesize")
            model = slot_values.get("fold_model", "quality_text_model")
            acc = slot_values.get("initial_value", "")
            for item in map_results:
                call_id = f"call_{len(trace.events):04d}"
                trace.record_call_start(call_id, model)
                r = await self.llm_provider.call(instruction=instruction, data=f"{acc}\n{item}", model=model)
                trace.record_call_end(call_id, tokens=r.tokens.total)
                acc = r.text
            return acc

        if map_results and not has_tree_reduce and not has_fold:
            return map_results

        # Iterate-until pattern
        if has_iterate:
            instruction = slot_values.get("instruction", "process")
            model = slot_values.get("model", "quality_text_model")
            max_iter = slot_values.get("max_iterations", 3)
            state: dict[str, Any] = {"error": "no attempt yet"}
            for i in range(max_iter):
                call_id = f"call_{len(trace.events):04d}"
                trace.record_call_start(call_id, model)
                r = await self.llm_provider.call(instruction=instruction, data=data, model=model)
                trace.record_call_end(call_id, tokens=r.tokens.total)
                state = {"result": r.text, "valid": True}
                break
            return state.get("result", "[no result]")

        # Simple direct call
        instruction = slot_values.get("instruction", "process")
        model = slot_values.get("model", "fast_text_model")
        call_id = f"call_{len(trace.events):04d}"
        trace.record_call_start(call_id, model)
        r = await self.llm_provider.call(instruction=instruction, data=data, model=model)
        trace.record_call_end(call_id, tokens=r.tokens.total)
        return r.text

    async def _tree_reduce(
        self,
        items: list[str],
        instruction: str,
        model: str,
        branch_factor: int,
        trace: TraceCollector,
    ) -> str:
        """Simulate tree reduction."""
        current = list(items)
        while len(current) > 1:
            groups: list[list[str]] = []
            for i in range(0, len(current), branch_factor):
                groups.append(current[i:i + branch_factor])
            next_level: list[str] = []
            for group in groups:
                combined = json.dumps(group)
                call_id = f"call_{len(trace.events):04d}"
                trace.record_call_start(call_id, model)
                r = await self.llm_provider.call(instruction=instruction, data=combined, model=model)
                trace.record_call_end(call_id, tokens=r.tokens.total)
                next_level.append(r.text)
            current = next_level
        return current[0] if current else ""


def _verify(
    artifact: Any,
    code: str,
    template: Template,
    policy: ExecutionPolicy,
    context_store: ContextStore,
    context_ids: list[str],
) -> VerificationRecord:
    """Run verification checks on an instantiated artifact."""
    checks: list[VerificationCheck] = []

    # 1. Hash integrity
    actual_hash = hashlib.sha256(code.encode()).hexdigest()
    checks.append(VerificationCheck(
        name="hash_match",
        status="pass" if actual_hash == artifact.generated_scheme_ref.hash else "fail",
        message="Code hash matches" if actual_hash == artifact.generated_scheme_ref.hash else "Hash mismatch",
    ))

    # 2. Only primitive bindings
    from .instantiator import ALLOWED_PRIMITIVES, FORBIDDEN_IDENTIFIERS
    tokens = set(re.findall(r"[a-zA-Z_][\w!?-]*", code))
    forbidden_found = tokens & FORBIDDEN_IDENTIFIERS
    checks.append(VerificationCheck(
        name="only_primitive_bindings",
        status="fail" if forbidden_found else "pass",
        message=f"Forbidden: {forbidden_found}" if forbidden_found else "All bindings allowed",
    ))

    # 3. Context exists
    for ctx_id in context_ids:
        rec = context_store.get(ctx_id)
        checks.append(VerificationCheck(
            name="context_id_exists",
            status="pass" if rec else "fail",
            message=f"Context {ctx_id} {'found' if rec else 'not found'}",
        ))

    # 4. Call count within policy
    profile = artifact.static_profile
    formula = profile.expected_calls_formula
    if formula and formula.isdigit():
        expected = int(formula)
        within = expected <= policy.max_llm_calls
        checks.append(VerificationCheck(
            name="expected_calls_within_policy",
            status="pass" if within else "warn",
            message=f"Expected {expected} calls (limit {policy.max_llm_calls})",
        ))

    # 5. Concurrency within policy
    if profile.max_concurrency:
        within = profile.max_concurrency <= policy.max_concurrency
        checks.append(VerificationCheck(
            name="max_concurrency_within_policy",
            status="pass" if within else "warn",
            message=f"Max concurrency {profile.max_concurrency} (limit {policy.max_concurrency})",
        ))

    # 6. Python bridge allowed
    if profile.uses_python_bridge and not policy.allow_python_bridge:
        checks.append(VerificationCheck(
            name="python_bridge_allowed",
            status="fail",
            message="Template uses Python bridge but policy disallows it",
        ))

    # 7. LLM-generated code allowed
    if template.uses_llm_generated_code and not policy.allow_llm_generated_code:
        checks.append(VerificationCheck(
            name="llm_generated_code_allowed",
            status="fail",
            message="Template uses LLM-generated code but policy disallows it",
        ))

    # Determine overall decision
    statuses = [c.status for c in checks]
    if "fail" in statuses:
        decision = VerificationDecision.fail
    elif "warn" in statuses:
        decision = VerificationDecision.warn
    else:
        decision = VerificationDecision.pass_

    return VerificationRecord(
        verification_id="pending",
        created_at=datetime.now(timezone.utc),
        artifact_id=artifact.artifact_id,
        decision=decision,
        checks=checks,
    )
