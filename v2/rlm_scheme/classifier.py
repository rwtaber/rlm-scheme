"""Deterministic TaskShape/DataShape classifier (Q0-Q9 decision tree)."""

from __future__ import annotations

from typing import Any

from .models import Classification, DataShape, TaskShape


def classify(
    task: str,
    metadata: dict[str, Any],
    hints: dict[str, Any] | None = None,
) -> Classification:
    """Classify a task + metadata into TaskShape and DataShape.

    Uses the Q0-Q9 decision tree from section 14.1 of the plan.
    Returns a Classification with confidence 1.0 for deterministic paths,
    or 0.5-0.8 for ambiguous cases that may need LLM gap-fill.
    """
    hints = hints or {}
    task_shape, confidence = _classify_task_shape(task, metadata, hints)
    data_shape = _classify_data_shape(metadata, hints)

    # Check for Composite (Q9)
    constituent_shapes = None
    if hints.get("multi_phase") or hints.get("phases"):
        task_shape = TaskShape.Composite
        constituent_shapes = hints.get("phases", [])
        if isinstance(constituent_shapes, list) and all(isinstance(s, str) for s in constituent_shapes):
            constituent_shapes = [TaskShape(s) for s in constituent_shapes if s in TaskShape.__members__]

    return Classification(
        task_shape=task_shape,
        data_shape=data_shape,
        confidence=confidence,
        constituent_shapes=constituent_shapes,
        rationale=_rationale(task_shape, data_shape, metadata),
    )


def _classify_task_shape(
    task: str,
    meta: dict[str, Any],
    hints: dict[str, Any],
) -> tuple[TaskShape, float]:
    """Q0-Q9 decision tree for TaskShape."""
    # Allow direct override via hints
    if "task_shape" in hints:
        try:
            return TaskShape(hints["task_shape"]), 1.0
        except ValueError:
            pass

    item_count = meta.get("item_count", 1) or 1
    independent = meta.get("independent", True)
    ordered = meta.get("ordered", False)
    has_second_phase = meta.get("has_second_phase") or hints.get("has_second_phase", False)

    # Q0: Single item, single operation, no second phase?
    # But only if no stronger signal from task keywords
    if item_count <= 1 and not has_second_phase:
        # Check Q5-Q8 keywords first before defaulting to Direct
        if _is_generation_task(task, hints):
            return TaskShape.Generate, 0.8
        if _is_refinement_task(task, hints):
            return TaskShape.Refine, 0.8
        if _is_decomposition_task(task, hints):
            return TaskShape.Decompose, 0.8
        if _is_comparison_task(task, hints):
            return TaskShape.Compare, 0.7
        return TaskShape.Direct, 1.0

    # Q1: Many input items?
    if item_count > 1:
        # Q2: Independent?
        if independent:
            # Q3: What per-item operation?
            op_type = _infer_operation_type(task, hints)
            if op_type == "classify":
                return TaskShape.Classify, 0.9
            elif op_type == "validate":
                return TaskShape.Validate, 0.9
            else:
                # Default: Batch (transform/extract)
                if has_second_phase:
                    return TaskShape.Composite, 0.9
                return TaskShape.Batch, 0.9
        else:
            # Q4: Ordered accumulation?
            if ordered:
                return TaskShape.Synthesize, 0.9
            return TaskShape.Pipeline, 0.8

    # Q5: Creating content, no source list?
    if _is_generation_task(task, hints):
        return TaskShape.Generate, 0.8

    # Q6: Improving one artifact?
    if _is_refinement_task(task, hints):
        return TaskShape.Refine, 0.8

    # Q7: Breaking input into parts?
    if _is_decomposition_task(task, hints):
        return TaskShape.Decompose, 0.8

    # Q8: Choosing among alternatives?
    if _is_comparison_task(task, hints):
        return TaskShape.Compare, 0.7

    # Default based on output
    output_type = meta.get("output_type") or hints.get("output_type", "one")
    if output_type == "many":
        return TaskShape.Aggregate, 0.5
    if has_second_phase:
        return TaskShape.Synthesize, 0.6

    return TaskShape.Direct, 0.5


def _classify_data_shape(meta: dict[str, Any], hints: dict[str, Any]) -> DataShape:
    """Determine DataShape from metadata."""
    if "data_shape" in hints:
        try:
            return DataShape(hints["data_shape"])
        except ValueError:
            pass

    if "data_shape" in meta and meta["data_shape"]:
        if isinstance(meta["data_shape"], DataShape):
            return meta["data_shape"]
        try:
            return DataShape(meta["data_shape"])
        except ValueError:
            pass

    item_count = meta.get("item_count", 1) or 1
    modality = meta.get("modality", [])

    if isinstance(modality, list) and any(m in ("image", "audio", "video") for m in modality):
        return DataShape.Multimodal

    if item_count > 1:
        if meta.get("tabular"):
            return DataShape.Tabular
        return DataShape.FlatList

    if meta.get("chunked") or meta.get("chunk_count"):
        return DataShape.ChunkedSingular

    if meta.get("hierarchical") or meta.get("depth", 0) > 2:
        return DataShape.Hierarchy

    return DataShape.Singular


def _infer_operation_type(task: str, hints: dict[str, Any]) -> str:
    """Infer per-item operation type from task description and hints."""
    if "operation" in hints:
        return hints["operation"]

    task_lower = task.lower()
    classify_words = {"classify", "categorize", "label", "tag", "sort into"}
    validate_words = {"validate", "check", "grade", "audit", "verify", "review"}

    for word in classify_words:
        if word in task_lower:
            return "classify"
    for word in validate_words:
        if word in task_lower:
            return "validate"
    return "extract"


def _is_generation_task(task: str, hints: dict[str, Any]) -> bool:
    if hints.get("generate"):
        return True
    words = {"generate", "create", "write", "compose", "draft", "produce"}
    return any(w in task.lower() for w in words)


def _is_refinement_task(task: str, hints: dict[str, Any]) -> bool:
    if hints.get("refine"):
        return True
    words = {"refine", "improve", "edit", "revise", "polish", "iterate"}
    return any(w in task.lower() for w in words)


def _is_decomposition_task(task: str, hints: dict[str, Any]) -> bool:
    if hints.get("decompose"):
        return True
    words = {"decompose", "break down", "split", "chunk", "partition"}
    return any(w in task.lower() for w in words)


def _is_comparison_task(task: str, hints: dict[str, Any]) -> bool:
    if hints.get("compare"):
        return True
    words = {"compare", "choose", "select", "pick", "rank", "evaluate alternatives"}
    return any(w in task.lower() for w in words)


def _rationale(ts: TaskShape, ds: DataShape, meta: dict[str, Any]) -> str:
    """Generate a brief rationale for the classification."""
    parts = [f"TaskShape={ts.value}"]
    if meta.get("item_count"):
        parts.append(f"item_count={meta['item_count']}")
    if meta.get("independent") is not None:
        parts.append(f"independent={meta['independent']}")
    parts.append(f"DataShape={ds.value}")
    return "; ".join(parts)
