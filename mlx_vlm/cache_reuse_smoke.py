import argparse
import json
import numbers
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache_boundary

from . import apc as _apc
from .generate import BatchGenerator, PromptCacheState, generate, normalize_resize_shape
from .prompt_utils import apply_chat_template
from .utils import load, prepare_inputs


@dataclass(frozen=True)
class PromptReusePlan:
    prompt_tokens: int
    reused_tokens: int
    used_boundary_restore: bool
    unsafe_rewind_refused: bool
    generated_tail_removed: bool
    recoverable: bool


@dataclass(frozen=True)
class CacheReuseSmokeReport:
    scenario: str
    model: str
    image: Optional[List[str]]
    prompt_tokens: int
    reused_tokens: int
    boundary_count: int
    main_cache_bytes: int
    boundary_cache_bytes: int
    total_cache_bytes: int
    peak_memory_bytes: Optional[int]
    image_token_id: Optional[int]
    image_token_in_prompt: bool
    image_token_in_suffix: bool
    first_wall_time_s: float
    second_wall_time_s: float
    used_boundary_restore: bool
    unsafe_rewind_refused: bool
    generated_tail_removed: bool
    recoverable: bool
    image_state_contract: str
    first_text: str
    second_text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CacheReuseParityReport:
    scenario: str
    model: str
    image: Optional[List[str]]
    top_k: int
    prompt_tokens: int
    reused_tokens: int
    boundary_count: int
    main_cache_bytes: int
    boundary_cache_bytes: int
    total_cache_bytes: int
    peak_memory_bytes: Optional[int]
    image_token_id: Optional[int]
    image_token_in_prompt: bool
    image_token_in_suffix: bool
    used_boundary_restore: bool
    unsafe_rewind_refused: bool
    generated_tail_removed: bool
    recoverable: bool
    cold_token: Optional[int]
    reused_token: Optional[int]
    tokens_match: bool
    parity: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CacheReuseBoundaryLedgerReport:
    scenario: str
    model: str
    image: Optional[List[str]]
    top_k: int
    prompt_tokens: int
    reused_tokens: int
    image_token_id: Optional[int]
    image_token_in_prompt: bool
    image_token_in_suffix: bool
    used_boundary_restore: bool
    unsafe_rewind_refused: bool
    generated_tail_removed: bool
    recoverable: bool
    cold_token: Optional[int]
    reused_token: Optional[int]
    tokens_match: bool
    parity: Dict[str, Any]
    ledger: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BatchCacheReuseSmokeReport:
    scenario: str
    model: str
    image: Optional[List[str]]
    image_token_id: Optional[int]
    first_prompt_tokens: List[int]
    second_prompt_tokens: List[int]
    prompt_progress: List[Dict[str, Any]]
    generated_token_counts: Dict[str, int]
    generated_texts: Dict[str, str]
    image_reuse_mode: str
    image_reused_tokens: int
    image_reuse_classification: Optional[Dict[str, Any]]
    text_reuse_mode: str
    text_reused_tokens: int
    apc_mode: Optional[str]
    peak_memory_bytes: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


BOUNDARY_LEDGER_EXACT_INVARIANTS = (
    "suffix_input_ids",
    "full_mask",
    "suffix_inputs_embeds",
    "suffix_position_ids",
    "image_grid_thw",
    "rope_deltas",
    "cold_prefix_cache_vs_first_boundary_cache",
    "first_boundary_cache_vs_recovered_cache",
)

EXPECTED_BOUNDARY_LEDGER_SHAPE_MISMATCHES = (
    "trimmed_suffix_mask_vs_reused_full_mask",
)


def cache_nbytes(value: Any) -> int:
    if value is None:
        return 0
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, numbers.Integral):
        return nbytes
    if isinstance(value, dict):
        return sum(cache_nbytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(cache_nbytes(v) for v in value)
    return 0


def _is_array(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "flatten")


def _array_preview(value: Any, limit: int = 8) -> List[Union[int, float, bool]]:
    if value is None:
        return []
    flat = value.flatten().tolist()
    return flat[: min(limit, len(flat))]


def array_summary(value: Any, *, preview_limit: int = 8) -> Dict[str, Any]:
    if value is None:
        return {"present": False}
    summary = {
        "present": True,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "nbytes": int(getattr(value, "nbytes", 0)),
        "preview": _array_preview(value, preview_limit),
    }
    if value.size > 0:
        try:
            summary["min"] = float(mx.min(value.astype(mx.float32)).item())
            summary["max"] = float(mx.max(value.astype(mx.float32)).item())
        except Exception:
            pass
    return summary


def array_comparison(cold: Any, reused: Any) -> Dict[str, Any]:
    if cold is None or reused is None:
        return {
            "cold_present": cold is not None,
            "reused_present": reused is not None,
            "shape_match": cold is None and reused is None,
            "exact_match": cold is None and reused is None,
            "max_abs_delta": None,
        }
    shape_match = tuple(cold.shape) == tuple(reused.shape)
    if not shape_match:
        return {
            "cold_present": True,
            "reused_present": True,
            "cold_shape": list(cold.shape),
            "reused_shape": list(reused.shape),
            "shape_match": False,
            "exact_match": False,
            "max_abs_delta": None,
        }
    exact_match = bool(mx.array_equal(cold, reused))
    max_abs_delta = None
    if cold.size > 0:
        max_abs_delta = float(
            mx.max(mx.abs(cold.astype(mx.float32) - reused.astype(mx.float32))).item()
        )
    return {
        "cold_present": True,
        "reused_present": True,
        "cold_shape": list(cold.shape),
        "reused_shape": list(reused.shape),
        "shape_match": True,
        "exact_match": exact_match,
        "max_abs_delta": max_abs_delta,
    }


def _flatten_arrays(value: Any) -> List[Any]:
    if value is None:
        return []
    if _is_array(value):
        return [value]
    state = getattr(value, "state", None)
    if state is not None and state is not value:
        return _flatten_arrays(state)
    if isinstance(value, dict):
        arrays = []
        for key in sorted(value):
            arrays.extend(_flatten_arrays(value[key]))
        return arrays
    if isinstance(value, (list, tuple)):
        arrays = []
        for item in value:
            arrays.extend(_flatten_arrays(item))
        return arrays
    return []


def cache_state_comparison(cold_cache: Any, reused_cache: Any) -> Dict[str, Any]:
    cold_arrays = _flatten_arrays(cold_cache)
    reused_arrays = _flatten_arrays(reused_cache)
    count_match = len(cold_arrays) == len(reused_arrays)
    comparisons = [
        array_comparison(cold, reused)
        for cold, reused in zip(cold_arrays, reused_arrays)
    ]
    exact_match = count_match and all(item["exact_match"] for item in comparisons)
    deltas = [
        item["max_abs_delta"]
        for item in comparisons
        if item["max_abs_delta"] is not None
    ]
    return {
        "cold_array_count": len(cold_arrays),
        "reused_array_count": len(reused_arrays),
        "array_count_match": count_match,
        "exact_match": exact_match,
        "max_abs_delta": max(deltas, default=None),
        "entries": comparisons,
    }


def classify_boundary_ledger_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize whether a boundary ledger found payload mismatch or drift only."""
    comparisons = report.get("ledger", {}).get("comparisons", {})
    failed_invariants = [
        name
        for name in BOUNDARY_LEDGER_EXACT_INVARIANTS
        if not comparisons.get(name, {}).get("exact_match", False)
    ]
    expected_shape_mismatches = [
        name
        for name in EXPECTED_BOUNDARY_LEDGER_SHAPE_MISMATCHES
        if name in comparisons
        and not comparisons.get(name, {}).get("shape_match", True)
    ]
    parity = report.get("parity", {})
    argmax_stable = bool(report.get("tokens_match", False)) and bool(
        parity.get("argmax_token_id_match", False)
    )
    distribution_exact = (
        argmax_stable
        and bool(parity.get("topk_token_ids_match", False))
        and parity.get("max_abs_logprob_delta", None) == 0.0
    )
    boundary_payload_exact = len(failed_invariants) == 0
    if not boundary_payload_exact:
        classification = "boundary_payload_mismatch"
    elif not argmax_stable:
        classification = "exact_boundary_payload_argmax_drift"
    elif distribution_exact:
        classification = "exact_boundary_payload_distribution_exact"
    else:
        classification = "exact_boundary_payload_distribution_drift"
    return {
        "boundary_payload_exact": boundary_payload_exact,
        "failed_boundary_invariants": failed_invariants,
        "expected_shape_mismatches": expected_shape_mismatches,
        "argmax_stable": argmax_stable,
        "distribution_exact": distribution_exact,
        "classification": classification,
        "live_confirmation_required": (
            boundary_payload_exact and not distribution_exact
        ),
    }


def _cache_offsets(prompt_cache: Any) -> List[Dict[str, Any]]:
    offsets = []
    for index, entry in enumerate(prompt_cache or []):
        offset = getattr(entry, "_idx", None)
        source = "_idx"
        if offset is None:
            offset = getattr(entry, "offset", None)
            source = "offset"
        offsets.append(
            {
                "index": index,
                "source": source if offset is not None else None,
                "value": _array_preview(offset) if _is_array(offset) else offset,
            }
        )
    return offsets


def _first_scalar_cache_offset(prompt_cache: Any) -> int:
    for entry in prompt_cache or []:
        offset = getattr(entry, "_idx", None)
        if offset is None:
            offset = getattr(entry, "offset", None)
        if offset is None:
            continue
        if _is_array(offset):
            flat = offset.flatten().tolist()
            return int(flat[0]) if flat else 0
        return int(offset)
    return 0


def _position_ids_for_boundary(
    *,
    model: Any,
    input_ids: Any,
    prompt_cache: Any,
    kwargs: Dict[str, Any],
) -> Any:
    lm = getattr(model, "language_model", None)
    if lm is None:
        return None
    cache_offset = _first_scalar_cache_offset(prompt_cache)
    position_ids = getattr(lm, "_position_ids", None)
    if (
        position_ids is not None
        and position_ids.ndim == 3
        and position_ids.shape[1] == input_ids.shape[0]
        and position_ids.shape[-1] >= cache_offset + input_ids.shape[1]
    ):
        return position_ids[:, :, cache_offset : cache_offset + input_ids.shape[1]]
    rope_deltas = kwargs.get("rope_deltas", None)
    if rope_deltas is None:
        rope_deltas = getattr(lm, "_rope_deltas", None)
    if rope_deltas is None:
        return None
    delta = mx.array(cache_offset + rope_deltas)
    if delta.ndim == 0:
        delta = mx.expand_dims(delta, axis=0)
    delta = delta.reshape(-1)[: input_ids.shape[0]]
    if delta.shape[0] < input_ids.shape[0]:
        delta = mx.tile(delta, (input_ids.shape[0],))[: input_ids.shape[0]]
    positions = mx.arange(input_ids.shape[1]).reshape(1, -1)
    positions = mx.add(positions, delta[:, None])[None, ...]
    return mx.broadcast_to(positions, (3, input_ids.shape[0], input_ids.shape[1]))


def _capture_boundary_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    kwargs = dict(payload["kwargs"])
    position_ids = _position_ids_for_boundary(
        model=payload["model"],
        input_ids=payload["input_ids"],
        prompt_cache=payload["prompt_cache"],
        kwargs=kwargs,
    )
    rope_deltas = kwargs.get("rope_deltas", None)
    if rope_deltas is None:
        rope_deltas = getattr(
            getattr(payload["model"], "language_model", None), "_rope_deltas", None
        )
    raw = {
        "input_ids": payload["input_ids"],
        "mask": payload["mask"],
        "pixel_values": payload["pixel_values"],
        "inputs_embeds": payload["inputs_embeds"],
        "position_ids": position_ids,
        "image_grid_thw": kwargs.get("image_grid_thw", None),
        "rope_deltas": rope_deltas,
        "prompt_cache": payload["prompt_cache"],
    }
    summary = {
        "input_ids": array_summary(raw["input_ids"]),
        "mask": array_summary(raw["mask"]),
        "pixel_values": array_summary(raw["pixel_values"]),
        "inputs_embeds": array_summary(raw["inputs_embeds"]),
        "position_ids": array_summary(raw["position_ids"]),
        "image_grid_thw": array_summary(raw["image_grid_thw"]),
        "rope_deltas": array_summary(raw["rope_deltas"]),
        "prompt_cache_bytes": cache_nbytes(raw["prompt_cache"]),
        "prompt_cache_offsets": _cache_offsets(raw["prompt_cache"]),
        "cached_image_features_present": kwargs.get("cached_image_features", None)
        is not None,
    }
    return summary, raw


def prompt_cache_state_metrics(state: PromptCacheState) -> Dict[str, int]:
    main_cache_bytes = cache_nbytes(state.cache)
    boundary_cache_bytes = cache_nbytes(state.boundary_cache)
    boundary_count = 1 if state.boundary_cache is not None else 0
    return {
        "boundary_count": boundary_count,
        "main_cache_bytes": main_cache_bytes,
        "boundary_cache_bytes": boundary_cache_bytes,
        "total_cache_bytes": main_cache_bytes + boundary_cache_bytes,
    }


def _logprob_values(logprobs: Any) -> List[float]:
    if logprobs is None:
        raise ValueError("Expected logprobs, got None.")
    if hasattr(logprobs, "flatten"):
        logprobs = logprobs.flatten().tolist()
    return [float(v) for v in logprobs]


def topk_logprob_snapshot(logprobs: Any, k: int) -> List[Dict[str, Union[int, float]]]:
    values = _logprob_values(logprobs)
    ranked = sorted(enumerate(values), key=lambda item: item[1], reverse=True)
    return [
        {"token_id": int(token_id), "logprob": float(logprob)}
        for token_id, logprob in ranked[:k]
    ]


def compare_topk_logprobs(cold_logprobs: Any, reused_logprobs: Any, k: int) -> Dict[str, Any]:
    cold_values = _logprob_values(cold_logprobs)
    reused_values = _logprob_values(reused_logprobs)
    if len(cold_values) != len(reused_values):
        raise ValueError(
            f"Logprob lengths differ: cold={len(cold_values)} reused={len(reused_values)}"
        )

    cold_topk = topk_logprob_snapshot(cold_values, k)
    reused_topk = topk_logprob_snapshot(reused_values, k)
    cold_ids = [entry["token_id"] for entry in cold_topk]
    reused_ids = [entry["token_id"] for entry in reused_topk]
    cold_argmax = cold_ids[0] if cold_ids else None
    reused_argmax = reused_ids[0] if reused_ids else None
    argmax_delta = (
        abs(float(cold_values[cold_argmax] - reused_values[cold_argmax]))
        if cold_argmax is not None and cold_argmax == reused_argmax
        else None
    )
    shared_ids = sorted(set(cold_ids) & set(reused_ids))
    shared_deltas = [
        {
            "token_id": int(token_id),
            "cold_logprob": float(cold_values[token_id]),
            "reused_logprob": float(reused_values[token_id]),
            "abs_delta": abs(float(cold_values[token_id] - reused_values[token_id])),
        }
        for token_id in shared_ids
    ]
    max_delta = max((entry["abs_delta"] for entry in shared_deltas), default=None)
    return {
        "cold_topk": cold_topk,
        "reused_topk": reused_topk,
        "cold_argmax_token_id": cold_argmax,
        "reused_argmax_token_id": reused_argmax,
        "argmax_token_id_match": cold_argmax == reused_argmax,
        "argmax_abs_logprob_delta": argmax_delta,
        "cold_topk_token_ids": cold_ids,
        "reused_topk_token_ids": reused_ids,
        "topk_token_ids_match": cold_ids == reused_ids,
        "shared_topk_token_count": len(shared_ids),
        "shared_topk_deltas": shared_deltas,
        "max_abs_logprob_delta": max_delta,
    }


def inspect_prompt_reuse(
    state: PromptCacheState, new_token_ids: Sequence[int]
) -> PromptReusePlan:
    new_token_ids = list(new_token_ids)
    prefix_len = state.find_prefix_length(new_token_ids)
    recovered = state.recover_prefix_cache(prefix_len)
    recoverable = recovered is not None
    boundary_match = (
        state.boundary_cache is not None
        and state.boundary_token_ids is not None
        and prefix_len == len(state.boundary_token_ids)
        and state.token_ids is not None
        and state.boundary_token_ids == state.token_ids[:prefix_len]
    )
    unsafe_rewind_refused = False
    if boundary_match and state.cache is not None and state.token_ids is not None:
        no_boundary = PromptCacheState()
        no_boundary.update(state.token_ids, state.cache)
        unsafe_rewind_refused = no_boundary.recover_prefix_cache(prefix_len) is None

    generated_tail_removed = (
        recoverable
        and boundary_match
        and state.token_ids is not None
        and prefix_len < len(state.token_ids)
    )
    return PromptReusePlan(
        prompt_tokens=len(new_token_ids),
        reused_tokens=prefix_len,
        used_boundary_restore=recoverable and boundary_match,
        unsafe_rewind_refused=unsafe_rewind_refused,
        generated_tail_removed=generated_tail_removed,
        recoverable=recoverable,
    )


def _as_list(value: Optional[Union[str, Sequence[str]]]) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return list(value)


def _config_dict(model: Any) -> Dict[str, Any]:
    config = getattr(model, "config", None)
    if config is None:
        return {}
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return dict(getattr(config, "__dict__", {}))


def _add_special_tokens(model: Any, processor: Any) -> bool:
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type in ["gemma3", "gemma3n", "gemma4"]:
        return getattr(processor, "chat_template", None) is None
    return True


def prepare_prompt_inputs(
    model: Any,
    processor: Any,
    prompt: str,
    *,
    image: Optional[List[str]],
    resize_shape: Optional[Union[int, Sequence[int]]],
    processor_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return prepare_inputs(
        processor,
        images=image,
        prompts=prompt,
        image_token_index=getattr(
            getattr(model, "config", None), "image_token_index", None
        ),
        resize_shape=normalize_resize_shape(resize_shape),
        add_special_tokens=_add_special_tokens(model, processor),
        **(processor_kwargs or {}),
    )


def append_suffix_tokens_to_inputs(
    inputs: Dict[str, Any], suffix_tokens: Sequence[int]
) -> Dict[str, Any]:
    suffix_tokens = list(suffix_tokens)
    if not suffix_tokens:
        return dict(inputs)

    extended = dict(inputs)
    input_ids = inputs["input_ids"]
    suffix = mx.array([suffix_tokens], dtype=input_ids.dtype)
    extended["input_ids"] = mx.concatenate([input_ids, suffix], axis=1)

    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        suffix_mask = mx.ones(
            (attention_mask.shape[0], len(suffix_tokens)),
            dtype=attention_mask.dtype,
        )
        extended["attention_mask"] = mx.concatenate(
            [attention_mask, suffix_mask], axis=1
        )
    return extended


def append_suffix_tokens_to_cached_turn_inputs(
    inputs: Dict[str, Any],
    *,
    cached_token_ids: Sequence[int],
    suffix_tokens: Sequence[int],
) -> Dict[str, Any]:
    token_ids = list(cached_token_ids) + list(suffix_tokens)
    extended = dict(inputs)
    extended["input_ids"] = mx.array([token_ids], dtype=inputs["input_ids"].dtype)

    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        extended["attention_mask"] = mx.ones(
            (attention_mask.shape[0], len(token_ids)),
            dtype=attention_mask.dtype,
        )
    return extended


def generation_kwargs_from_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    kwargs = {
        k: v
        for k, v in inputs.items()
        if k not in ["input_ids", "pixel_values", "attention_mask"]
    }
    kwargs["input_ids"] = inputs["input_ids"]
    kwargs["pixel_values"] = inputs.get("pixel_values", None)
    kwargs["mask"] = inputs.get("attention_mask", None)
    return kwargs


def batch_generator_prompt_kwargs(
    model: Any,
    inputs: Dict[str, Any],
    *,
    apc_image_hash: Optional[int] = None,
    cache_reuse_live_marker: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    data_kwargs = {
        k: v
        for k, v in inputs.items()
        if k not in ["input_ids", "pixel_values", "attention_mask"]
    }
    language_model = getattr(model, "language_model", None)
    if language_model is not None:
        if hasattr(language_model, "_position_ids"):
            language_model._position_ids = None
        if hasattr(language_model, "_rope_deltas"):
            language_model._rope_deltas = None
    embedding_output = model.get_input_embeddings(
        inputs["input_ids"],
        inputs.get("pixel_values", None),
        mask=inputs.get("attention_mask", None),
        **data_kwargs,
    )
    prompt_kwargs = {**data_kwargs, **embedding_output.to_dict()}
    if language_model is not None:
        position_ids = getattr(language_model, "_position_ids", None)
        if (
            position_ids is not None
            and getattr(position_ids, "ndim", 0) >= 3
            and position_ids.shape[-1] == inputs["input_ids"].shape[-1]
        ):
            prompt_kwargs["position_ids"] = position_ids
        else:
            token_positions = mx.arange(inputs["input_ids"].shape[-1], dtype=mx.int32)
            token_positions = token_positions.reshape(1, 1, -1)
            prompt_kwargs["position_ids"] = mx.broadcast_to(
                token_positions,
                (3, 1, inputs["input_ids"].shape[-1]),
            )
        rope_deltas = getattr(language_model, "_rope_deltas", None)
        if rope_deltas is not None:
            prompt_kwargs["rope_deltas"] = rope_deltas
    if apc_image_hash is not None:
        prompt_kwargs["_apc_image_hash"] = int(apc_image_hash)
    if cache_reuse_live_marker is not None:
        prompt_kwargs["_cache_reuse_live_marker"] = dict(cache_reuse_live_marker)
    return prompt_kwargs


def _decode_tokens(processor: Any, tokens: Sequence[int]) -> str:
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    detokenizer = getattr(processor, "detokenizer", None)
    if detokenizer is not None:
        detokenizer.reset()
        for token in tokens:
            detokenizer.add_token(int(token))
        detokenizer.finalize()
        return detokenizer.text
    return tokenizer.decode(list(tokens))


def run_batch_generator_to_completion(
    *,
    model: Any,
    processor: Any,
    input_ids: List[List[int]],
    prompt_kwargs: List[Dict[str, Any]],
    max_tokens: int,
    apc_manager: "_apc.APCManager",
) -> Tuple[List[Dict[str, Any]], Dict[int, List[int]], Optional[str]]:
    language_model = getattr(model, "language_model", model)
    if hasattr(language_model, "_position_ids"):
        language_model._position_ids = None
    if hasattr(language_model, "_rope_deltas"):
        language_model._rope_deltas = None
    generator = BatchGenerator(
        language_model,
        processor,
        prefill_batch_size=len(input_ids),
        completion_batch_size=len(input_ids),
        compute_logprobs=False,
        apc_manager=apc_manager,
    )
    uids = generator.insert(
        input_ids,
        [max_tokens] * len(input_ids),
        prompt_kwargs=prompt_kwargs,
    )
    generated = {int(uid): [] for uid in uids}
    prompt_progress: List[Dict[str, Any]] = []
    try:
        while generator.has_work:
            prompt_responses, generation_responses = generator.next()
            for response in prompt_responses:
                row = asdict(response)
                row["uid"] = int(row["uid"])
                prompt_progress.append(row)
            for response in generation_responses:
                if response.finish_reason != "stop":
                    generated[int(response.uid)].append(_token_value(response.token))
        apc_mode = getattr(generator, "apc_mode", None)
    finally:
        generator.close()
    return prompt_progress, generated, apc_mode


def _image_state_contract(model: Any, prompt: str, plan: PromptReusePlan) -> str:
    image_token_id = getattr(getattr(model, "config", None), "image_token_id", None)
    image_token_id = image_token_id or getattr(
        getattr(model, "config", None), "image_token_index", None
    )
    if image_token_id is None:
        return "no_image_token_id"
    if not plan.recoverable:
        return "not_recovered"
    return "reprimed_or_not_in_suffix"


def _image_token_id(model: Any) -> Optional[int]:
    image_token_id = getattr(getattr(model, "config", None), "image_token_id", None)
    return image_token_id or getattr(
        getattr(model, "config", None), "image_token_index", None
    )


def run_image_prefix_smoke(
    *,
    model_path: str,
    image: Union[str, Sequence[str]],
    prefix_prompt: str,
    second_suffix: str,
    max_tokens: int,
    resize_shape: Optional[Union[int, Sequence[int]]] = None,
    trust_remote_code: bool = False,
    processor_kwargs: Optional[Dict[str, Any]] = None,
) -> CacheReuseSmokeReport:
    images = _as_list(image)
    model, processor = load(model_path, trust_remote_code=trust_remote_code)
    config = _config_dict(model)
    num_images = len(images or [])
    first_prompt = apply_chat_template(
        processor, config, prefix_prompt, num_images=num_images
    )
    first_inputs = prepare_prompt_inputs(
        model,
        processor,
        first_prompt,
        image=images,
        resize_shape=resize_shape,
        processor_kwargs=processor_kwargs,
    )
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    suffix_tokens = tokenizer.encode(second_suffix, add_special_tokens=False)
    second_inputs = append_suffix_tokens_to_inputs(first_inputs, suffix_tokens)
    first_ids = first_inputs["input_ids"].flatten().tolist()
    second_ids = second_inputs["input_ids"].flatten().tolist()
    image_token_id = _image_token_id(model)

    first_kwargs = generation_kwargs_from_inputs(first_inputs)
    second_kwargs = generation_kwargs_from_inputs(second_inputs)

    second_prompt_label = f"{first_prompt}{second_suffix}"

    state = PromptCacheState()
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    first_start = time.perf_counter()
    first_result = generate(
        model,
        processor,
        "",
        image=None,
        max_tokens=max_tokens,
        resize_shape=resize_shape,
        prompt_cache_state=state,
        **first_kwargs,
    )
    first_wall_time = time.perf_counter() - first_start

    plan = inspect_prompt_reuse(state, second_ids)

    second_start = time.perf_counter()
    second_result = generate(
        model,
        processor,
        "",
        image=None,
        max_tokens=max_tokens,
        resize_shape=resize_shape,
        prompt_cache_state=state,
        **second_kwargs,
    )
    second_wall_time = time.perf_counter() - second_start

    metrics = prompt_cache_state_metrics(state)
    peak_memory_bytes = (
        int(mx.get_peak_memory()) if hasattr(mx, "get_peak_memory") else None
    )
    return CacheReuseSmokeReport(
        scenario="image_prefix_diverged_suffix",
        model=model_path,
        image=images,
        prompt_tokens=plan.prompt_tokens,
        reused_tokens=plan.reused_tokens,
        boundary_count=metrics["boundary_count"],
        main_cache_bytes=metrics["main_cache_bytes"],
        boundary_cache_bytes=metrics["boundary_cache_bytes"],
        total_cache_bytes=metrics["total_cache_bytes"],
        peak_memory_bytes=peak_memory_bytes,
        image_token_id=image_token_id,
        image_token_in_prompt=(
            image_token_id is not None and image_token_id in first_ids
        ),
        image_token_in_suffix=(
            image_token_id is not None and image_token_id in suffix_tokens
        ),
        first_wall_time_s=first_wall_time,
        second_wall_time_s=second_wall_time,
        used_boundary_restore=plan.used_boundary_restore,
        unsafe_rewind_refused=plan.unsafe_rewind_refused,
        generated_tail_removed=plan.generated_tail_removed,
        recoverable=plan.recoverable,
        image_state_contract=_image_state_contract(model, second_prompt_label, plan),
        first_text=first_result.text,
        second_text=second_result.text,
    )


def run_image_multiturn_smoke(
    *,
    model_path: str,
    image: Union[str, Sequence[str]],
    prefix_prompt: str,
    second_suffix: str,
    max_tokens: int,
    resize_shape: Optional[Union[int, Sequence[int]]] = None,
    trust_remote_code: bool = False,
    processor_kwargs: Optional[Dict[str, Any]] = None,
) -> CacheReuseSmokeReport:
    images = _as_list(image)
    model, processor = load(model_path, trust_remote_code=trust_remote_code)
    config = _config_dict(model)
    num_images = len(images or [])
    first_prompt = apply_chat_template(
        processor, config, prefix_prompt, num_images=num_images
    )
    first_inputs = prepare_prompt_inputs(
        model,
        processor,
        first_prompt,
        image=images,
        resize_shape=resize_shape,
        processor_kwargs=processor_kwargs,
    )
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    suffix_tokens = tokenizer.encode(second_suffix, add_special_tokens=False)
    first_ids = first_inputs["input_ids"].flatten().tolist()
    image_token_id = _image_token_id(model)

    first_kwargs = generation_kwargs_from_inputs(first_inputs)

    state = PromptCacheState()
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    first_start = time.perf_counter()
    first_result = generate(
        model,
        processor,
        "",
        image=None,
        max_tokens=max_tokens,
        resize_shape=resize_shape,
        prompt_cache_state=state,
        **first_kwargs,
    )
    first_wall_time = time.perf_counter() - first_start

    if state.token_ids is None:
        raise RuntimeError("Prompt cache state did not capture the first turn.")

    second_inputs = append_suffix_tokens_to_cached_turn_inputs(
        first_inputs,
        cached_token_ids=state.token_ids,
        suffix_tokens=suffix_tokens,
    )
    second_ids = second_inputs["input_ids"].flatten().tolist()
    second_kwargs = generation_kwargs_from_inputs(second_inputs)
    plan = inspect_prompt_reuse(state, second_ids)

    second_start = time.perf_counter()
    second_result = generate(
        model,
        processor,
        "",
        image=None,
        max_tokens=max_tokens,
        resize_shape=resize_shape,
        prompt_cache_state=state,
        **second_kwargs,
    )
    second_wall_time = time.perf_counter() - second_start

    metrics = prompt_cache_state_metrics(state)
    peak_memory_bytes = (
        int(mx.get_peak_memory()) if hasattr(mx, "get_peak_memory") else None
    )
    return CacheReuseSmokeReport(
        scenario="image_multiturn_text_followup",
        model=model_path,
        image=images,
        prompt_tokens=plan.prompt_tokens,
        reused_tokens=plan.reused_tokens,
        boundary_count=metrics["boundary_count"],
        main_cache_bytes=metrics["main_cache_bytes"],
        boundary_cache_bytes=metrics["boundary_cache_bytes"],
        total_cache_bytes=metrics["total_cache_bytes"],
        peak_memory_bytes=peak_memory_bytes,
        image_token_id=image_token_id,
        image_token_in_prompt=(
            image_token_id is not None and image_token_id in first_ids
        ),
        image_token_in_suffix=(
            image_token_id is not None and image_token_id in suffix_tokens
        ),
        first_wall_time_s=first_wall_time,
        second_wall_time_s=second_wall_time,
        used_boundary_restore=plan.used_boundary_restore,
        unsafe_rewind_refused=plan.unsafe_rewind_refused,
        generated_tail_removed=plan.generated_tail_removed,
        recoverable=plan.recoverable,
        image_state_contract=_image_state_contract(model, second_suffix, plan),
        first_text=first_result.text,
        second_text=second_result.text,
    )


def run_image_prefix_batched_smoke(
    *,
    model_path: str,
    image: Union[str, Sequence[str]],
    prefix_prompt: str,
    second_suffix: str,
    max_tokens: int,
    resize_shape: Optional[Union[int, Sequence[int]]] = None,
    trust_remote_code: bool = False,
    processor_kwargs: Optional[Dict[str, Any]] = None,
) -> BatchCacheReuseSmokeReport:
    images = _as_list(image)
    model, processor = load(model_path, trust_remote_code=trust_remote_code)
    config = _config_dict(model)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    image_token_id = _image_token_id(model)
    apc_manager = _apc.APCManager()

    image_first_prompt = apply_chat_template(
        processor, config, prefix_prompt, num_images=len(images or [])
    )
    text_first_prompt = apply_chat_template(
        processor,
        config,
        "State one short fact about cache reuse.",
        num_images=0,
    )
    image_first_inputs = prepare_prompt_inputs(
        model,
        processor,
        image_first_prompt,
        image=images,
        resize_shape=resize_shape,
        processor_kwargs=processor_kwargs,
    )
    text_first_inputs = prepare_prompt_inputs(
        model,
        processor,
        text_first_prompt,
        image=None,
        resize_shape=resize_shape,
        processor_kwargs=processor_kwargs,
    )
    image_hash = _apc.hash_image_payload(
        pixel_values=image_first_inputs.get("pixel_values"),
        image_ref=images[0] if images else None,
    )

    first_input_ids = [
        image_first_inputs["input_ids"].flatten().tolist(),
        text_first_inputs["input_ids"].flatten().tolist(),
    ]
    first_prompt_kwargs = [
        batch_generator_prompt_kwargs(
            model,
            image_first_inputs,
            apc_image_hash=image_hash,
        ),
        batch_generator_prompt_kwargs(model, text_first_inputs),
    ]

    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    run_batch_generator_to_completion(
        model=model,
        processor=processor,
        input_ids=first_input_ids,
        prompt_kwargs=first_prompt_kwargs,
        max_tokens=max_tokens,
        apc_manager=apc_manager,
    )

    suffix_tokens = tokenizer.encode(second_suffix, add_special_tokens=False)
    text_suffix_tokens = tokenizer.encode(
        "\nNow answer with one different short fact.",
        add_special_tokens=False,
    )
    image_second_inputs = append_suffix_tokens_to_inputs(
        image_first_inputs,
        suffix_tokens,
    )
    text_second_inputs = append_suffix_tokens_to_inputs(
        text_first_inputs,
        text_suffix_tokens,
    )
    image_marker = {
        "classification": "exact_boundary_payload_distribution_drift",
        "boundary_payload_exact": True,
        "live_confirmation_required": True,
        "source": "cash_money_image_prefix_boundary_ledger_contract",
    }
    second_input_ids = [
        image_second_inputs["input_ids"].flatten().tolist(),
        text_second_inputs["input_ids"].flatten().tolist(),
    ]
    second_prompt_kwargs = [
        batch_generator_prompt_kwargs(
            model,
            image_second_inputs,
            apc_image_hash=image_hash,
            cache_reuse_live_marker=image_marker,
        ),
        batch_generator_prompt_kwargs(model, text_second_inputs),
    ]
    prompt_progress, generated, apc_mode = run_batch_generator_to_completion(
        model=model,
        processor=processor,
        input_ids=second_input_ids,
        prompt_kwargs=second_prompt_kwargs,
        max_tokens=max_tokens,
        apc_manager=apc_manager,
    )
    progress_by_uid = {int(row["uid"]): row for row in prompt_progress}
    if 0 not in progress_by_uid or 1 not in progress_by_uid:
        raise RuntimeError("Batched smoke did not report prompt progress for both rows.")
    image_progress = progress_by_uid[0]
    text_progress = progress_by_uid[1]
    if image_progress.get("reused_tokens", 0) <= 0:
        raise RuntimeError("Image row did not reuse cached prefix tokens.")
    if text_progress.get("reused_tokens", 0) <= 0:
        raise RuntimeError("Text row did not reuse cached prefix tokens.")
    if image_progress.get("reuse_classification") is None:
        raise RuntimeError("Image row did not preserve reuse classification marker.")

    generated_texts = {
        str(uid): _decode_tokens(processor, tokens)
        for uid, tokens in sorted(generated.items())
    }
    peak_memory_bytes = (
        int(mx.get_peak_memory()) if hasattr(mx, "get_peak_memory") else None
    )
    return BatchCacheReuseSmokeReport(
        scenario="image_prefix_batched_cache_reuse",
        model=model_path,
        image=images,
        image_token_id=image_token_id,
        first_prompt_tokens=[len(ids) for ids in first_input_ids],
        second_prompt_tokens=[len(ids) for ids in second_input_ids],
        prompt_progress=prompt_progress,
        generated_token_counts={
            str(uid): len(tokens) for uid, tokens in sorted(generated.items())
        },
        generated_texts=generated_texts,
        image_reuse_mode=str(image_progress.get("reuse_mode", "")),
        image_reused_tokens=int(image_progress.get("reused_tokens", 0) or 0),
        image_reuse_classification=image_progress.get("reuse_classification"),
        text_reuse_mode=str(text_progress.get("reuse_mode", "")),
        text_reused_tokens=int(text_progress.get("reused_tokens", 0) or 0),
        apc_mode=apc_mode,
        peak_memory_bytes=peak_memory_bytes,
    )


def _token_value(token: Any) -> Optional[int]:
    if token is None:
        return None
    if hasattr(token, "item"):
        return int(token.item())
    return int(token)


def run_image_prefix_parity(
    *,
    model_path: str,
    image: Union[str, Sequence[str]],
    prefix_prompt: str,
    second_suffix: str,
    top_k: int,
    resize_shape: Optional[Union[int, Sequence[int]]] = None,
    trust_remote_code: bool = False,
    processor_kwargs: Optional[Dict[str, Any]] = None,
    parity_order: str = "cold-first",
) -> CacheReuseParityReport:
    images = _as_list(image)
    model, processor = load(model_path, trust_remote_code=trust_remote_code)
    config = _config_dict(model)
    num_images = len(images or [])
    first_prompt = apply_chat_template(
        processor, config, prefix_prompt, num_images=num_images
    )
    first_inputs = prepare_prompt_inputs(
        model,
        processor,
        first_prompt,
        image=images,
        resize_shape=resize_shape,
        processor_kwargs=processor_kwargs,
    )
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    suffix_tokens = tokenizer.encode(second_suffix, add_special_tokens=False)
    second_inputs = append_suffix_tokens_to_inputs(first_inputs, suffix_tokens)
    first_ids = first_inputs["input_ids"].flatten().tolist()
    second_ids = second_inputs["input_ids"].flatten().tolist()
    image_token_id = _image_token_id(model)

    state = PromptCacheState()
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    generate(
        model,
        processor,
        "",
        image=None,
        max_tokens=1,
        resize_shape=resize_shape,
        prompt_cache_state=state,
        **generation_kwargs_from_inputs(first_inputs),
    )

    plan = inspect_prompt_reuse(state, second_ids)
    if parity_order == "cold-first":
        cold_result = generate(
            model,
            processor,
            "",
            image=None,
            max_tokens=1,
            resize_shape=resize_shape,
            **generation_kwargs_from_inputs(second_inputs),
        )
        reused_result = generate(
            model,
            processor,
            "",
            image=None,
            max_tokens=1,
            resize_shape=resize_shape,
            prompt_cache_state=state,
            **generation_kwargs_from_inputs(second_inputs),
        )
    elif parity_order == "reused-first":
        reused_result = generate(
            model,
            processor,
            "",
            image=None,
            max_tokens=1,
            resize_shape=resize_shape,
            prompt_cache_state=state,
            **generation_kwargs_from_inputs(second_inputs),
        )
        cold_result = generate(
            model,
            processor,
            "",
            image=None,
            max_tokens=1,
            resize_shape=resize_shape,
            **generation_kwargs_from_inputs(second_inputs),
        )
    else:
        raise ValueError(f"Unsupported parity order: {parity_order}")

    metrics = prompt_cache_state_metrics(state)
    peak_memory_bytes = (
        int(mx.get_peak_memory()) if hasattr(mx, "get_peak_memory") else None
    )
    cold_token = _token_value(cold_result.token)
    reused_token = _token_value(reused_result.token)
    return CacheReuseParityReport(
        scenario=f"image_prefix_diverged_suffix_parity_{parity_order}",
        model=model_path,
        image=images,
        top_k=top_k,
        prompt_tokens=plan.prompt_tokens,
        reused_tokens=plan.reused_tokens,
        boundary_count=metrics["boundary_count"],
        main_cache_bytes=metrics["main_cache_bytes"],
        boundary_cache_bytes=metrics["boundary_cache_bytes"],
        total_cache_bytes=metrics["total_cache_bytes"],
        peak_memory_bytes=peak_memory_bytes,
        image_token_id=image_token_id,
        image_token_in_prompt=(
            image_token_id is not None and image_token_id in first_ids
        ),
        image_token_in_suffix=(
            image_token_id is not None and image_token_id in suffix_tokens
        ),
        used_boundary_restore=plan.used_boundary_restore,
        unsafe_rewind_refused=plan.unsafe_rewind_refused,
        generated_tail_removed=plan.generated_tail_removed,
        recoverable=plan.recoverable,
        cold_token=cold_token,
        reused_token=reused_token,
        tokens_match=cold_token == reused_token,
        parity=compare_topk_logprobs(
            cold_result.logprobs, reused_result.logprobs, top_k
        ),
    )


def run_image_prefix_boundary_only_parity(
    *,
    model_path: str,
    image: Union[str, Sequence[str]],
    prefix_prompt: str,
    second_suffix: str,
    top_k: int,
    resize_shape: Optional[Union[int, Sequence[int]]] = None,
    trust_remote_code: bool = False,
    processor_kwargs: Optional[Dict[str, Any]] = None,
    parity_order: str = "cold-first",
) -> CacheReuseParityReport:
    images = _as_list(image)
    model, processor = load(model_path, trust_remote_code=trust_remote_code)
    config = _config_dict(model)
    num_images = len(images or [])
    first_prompt = apply_chat_template(
        processor, config, prefix_prompt, num_images=num_images
    )
    first_inputs = prepare_prompt_inputs(
        model,
        processor,
        first_prompt,
        image=images,
        resize_shape=resize_shape,
        processor_kwargs=processor_kwargs,
    )
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    suffix_tokens = tokenizer.encode(second_suffix, add_special_tokens=False)
    second_inputs = append_suffix_tokens_to_inputs(first_inputs, suffix_tokens)
    first_ids = first_inputs["input_ids"].flatten().tolist()
    second_ids = second_inputs["input_ids"].flatten().tolist()
    image_token_id = _image_token_id(model)

    initial_state = PromptCacheState()
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    generate(
        model,
        processor,
        "",
        image=None,
        max_tokens=1,
        resize_shape=resize_shape,
        prompt_cache_state=initial_state,
        **generation_kwargs_from_inputs(first_inputs),
    )
    if initial_state.boundary_cache is None or initial_state.boundary_token_ids is None:
        raise RuntimeError("Prompt cache state did not capture a prompt boundary.")

    state = PromptCacheState()
    state.update(initial_state.boundary_token_ids, initial_state.boundary_cache)

    plan = inspect_prompt_reuse(state, second_ids)
    if parity_order == "cold-first":
        cold_result = generate(
            model,
            processor,
            "",
            image=None,
            max_tokens=1,
            resize_shape=resize_shape,
            **generation_kwargs_from_inputs(second_inputs),
        )
        reused_result = generate(
            model,
            processor,
            "",
            image=None,
            max_tokens=1,
            resize_shape=resize_shape,
            prompt_cache_state=state,
            **generation_kwargs_from_inputs(second_inputs),
        )
    elif parity_order == "reused-first":
        reused_result = generate(
            model,
            processor,
            "",
            image=None,
            max_tokens=1,
            resize_shape=resize_shape,
            prompt_cache_state=state,
            **generation_kwargs_from_inputs(second_inputs),
        )
        cold_result = generate(
            model,
            processor,
            "",
            image=None,
            max_tokens=1,
            resize_shape=resize_shape,
            **generation_kwargs_from_inputs(second_inputs),
        )
    else:
        raise ValueError(f"Unsupported parity order: {parity_order}")

    metrics = prompt_cache_state_metrics(state)
    peak_memory_bytes = (
        int(mx.get_peak_memory()) if hasattr(mx, "get_peak_memory") else None
    )
    cold_token = _token_value(cold_result.token)
    reused_token = _token_value(reused_result.token)
    return CacheReuseParityReport(
        scenario=f"image_prefix_boundary_only_parity_{parity_order}",
        model=model_path,
        image=images,
        top_k=top_k,
        prompt_tokens=plan.prompt_tokens,
        reused_tokens=plan.reused_tokens,
        boundary_count=metrics["boundary_count"],
        main_cache_bytes=metrics["main_cache_bytes"],
        boundary_cache_bytes=metrics["boundary_cache_bytes"],
        total_cache_bytes=metrics["total_cache_bytes"],
        peak_memory_bytes=peak_memory_bytes,
        image_token_id=image_token_id,
        image_token_in_prompt=(
            image_token_id is not None and image_token_id in first_ids
        ),
        image_token_in_suffix=(
            image_token_id is not None and image_token_id in suffix_tokens
        ),
        used_boundary_restore=plan.used_boundary_restore,
        unsafe_rewind_refused=plan.unsafe_rewind_refused,
        generated_tail_removed=plan.generated_tail_removed,
        recoverable=plan.recoverable,
        cold_token=cold_token,
        reused_token=reused_token,
        tokens_match=cold_token == reused_token,
        parity=compare_topk_logprobs(
            cold_result.logprobs, reused_result.logprobs, top_k
        ),
    )


def run_image_prefix_boundary_ledger(
    *,
    model_path: str,
    image: Union[str, Sequence[str]],
    prefix_prompt: str,
    second_suffix: str,
    top_k: int,
    resize_shape: Optional[Union[int, Sequence[int]]] = None,
    trust_remote_code: bool = False,
    processor_kwargs: Optional[Dict[str, Any]] = None,
    parity_order: str = "cold-first",
) -> CacheReuseBoundaryLedgerReport:
    images = _as_list(image)
    model, processor = load(model_path, trust_remote_code=trust_remote_code)
    config = _config_dict(model)
    num_images = len(images or [])
    first_prompt = apply_chat_template(
        processor, config, prefix_prompt, num_images=num_images
    )
    first_inputs = prepare_prompt_inputs(
        model,
        processor,
        first_prompt,
        image=images,
        resize_shape=resize_shape,
        processor_kwargs=processor_kwargs,
    )
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    suffix_tokens = tokenizer.encode(second_suffix, add_special_tokens=False)
    second_inputs = append_suffix_tokens_to_inputs(first_inputs, suffix_tokens)
    first_ids = first_inputs["input_ids"].flatten().tolist()
    second_ids = second_inputs["input_ids"].flatten().tolist()
    image_token_id = _image_token_id(model)

    initial_state = PromptCacheState()
    generate(
        model,
        processor,
        "",
        image=None,
        max_tokens=1,
        resize_shape=resize_shape,
        prompt_cache_state=initial_state,
        **generation_kwargs_from_inputs(first_inputs),
    )
    if initial_state.boundary_cache is None or initial_state.boundary_token_ids is None:
        raise RuntimeError("Prompt cache state did not capture a prompt boundary.")

    state = PromptCacheState()
    state.update(initial_state.boundary_token_ids, initial_state.boundary_cache)
    plan = inspect_prompt_reuse(state, second_ids)

    cold_boundary: Dict[str, Any] = {}
    reused_boundary: Dict[str, Any] = {}
    cold_raw: Dict[str, Any] = {}
    reused_raw: Dict[str, Any] = {}
    cold_prefix_cache: List[Any] = []

    def capture_cold_boundary(payload: Dict[str, Any]) -> None:
        nonlocal cold_boundary, cold_raw
        cold_boundary, cold_raw = _capture_boundary_payload(payload)

    def capture_reused_boundary(payload: Dict[str, Any]) -> None:
        nonlocal reused_boundary, reused_raw
        reused_boundary, reused_raw = _capture_boundary_payload(payload)

    def capture_cold_prefix_cache(prefix_len: int, prompt_cache: List[Any]) -> None:
        nonlocal cold_prefix_cache
        cold_prefix_cache = make_prompt_cache_boundary(prompt_cache)

    cold_kwargs = generation_kwargs_from_inputs(second_inputs)
    cold_kwargs["prompt_boundary_ledger_callback"] = capture_cold_boundary
    cold_kwargs["prompt_cache_checkpoint"] = capture_cold_prefix_cache
    cold_kwargs["prompt_cache_checkpoint_len"] = plan.reused_tokens
    reused_kwargs = generation_kwargs_from_inputs(second_inputs)
    reused_kwargs["prompt_boundary_ledger_callback"] = capture_reused_boundary
    reused_kwargs["prompt_cache_state"] = state
    recovered_prefix_cache = state.recover_prefix_cache(plan.reused_tokens)

    if parity_order == "cold-first":
        cold_result = generate(
            model,
            processor,
            "",
            image=None,
            max_tokens=1,
            resize_shape=resize_shape,
            **cold_kwargs,
        )
        reused_result = generate(
            model,
            processor,
            "",
            image=None,
            max_tokens=1,
            resize_shape=resize_shape,
            **reused_kwargs,
        )
    elif parity_order == "reused-first":
        reused_result = generate(
            model,
            processor,
            "",
            image=None,
            max_tokens=1,
            resize_shape=resize_shape,
            **reused_kwargs,
        )
        cold_result = generate(
            model,
            processor,
            "",
            image=None,
            max_tokens=1,
            resize_shape=resize_shape,
            **cold_kwargs,
        )
    else:
        raise ValueError(f"Unsupported parity order: {parity_order}")

    if not cold_boundary or not reused_boundary:
        raise RuntimeError("Boundary ledger callback did not capture both paths.")

    cold_suffix = {
        "input_ids": cold_raw["input_ids"][:, plan.reused_tokens :],
        "mask": (
            cold_raw["mask"][:, plan.reused_tokens :]
            if cold_raw["mask"] is not None
            else None
        ),
        "inputs_embeds": cold_raw["inputs_embeds"][:, plan.reused_tokens :, :],
        "position_ids": (
            cold_raw["position_ids"][:, :, plan.reused_tokens :]
            if cold_raw["position_ids"] is not None
            else None
        ),
    }
    prefix_cache_comparison = cache_state_comparison(
        cold_prefix_cache,
        initial_state.boundary_cache,
    )
    restored_cache_comparison = cache_state_comparison(
        initial_state.boundary_cache,
        recovered_prefix_cache,
    )
    comparisons = {
        "full_mask": array_comparison(cold_raw["mask"], reused_raw["mask"]),
        "suffix_input_ids": array_comparison(
            cold_suffix["input_ids"], reused_raw["input_ids"]
        ),
        "trimmed_suffix_mask_vs_reused_full_mask": array_comparison(
            cold_suffix["mask"], reused_raw["mask"]
        ),
        "suffix_inputs_embeds": array_comparison(
            cold_suffix["inputs_embeds"], reused_raw["inputs_embeds"]
        ),
        "suffix_position_ids": array_comparison(
            cold_suffix["position_ids"], reused_raw["position_ids"]
        ),
        "image_grid_thw": array_comparison(
            cold_raw["image_grid_thw"], reused_raw["image_grid_thw"]
        ),
        "rope_deltas": array_comparison(
            cold_raw["rope_deltas"], reused_raw["rope_deltas"]
        ),
        "cold_prefix_cache_vs_first_boundary_cache": prefix_cache_comparison,
        "first_boundary_cache_vs_recovered_cache": restored_cache_comparison,
    }

    cold_token = _token_value(cold_result.token)
    reused_token = _token_value(reused_result.token)
    parity = compare_topk_logprobs(cold_result.logprobs, reused_result.logprobs, top_k)
    ledger = {
        "cold_boundary": cold_boundary,
        "reused_boundary": reused_boundary,
        "cold_suffix": {k: array_summary(v) for k, v in cold_suffix.items()},
        "comparisons": comparisons,
    }
    ledger["classification"] = classify_boundary_ledger_report(
        {
            "tokens_match": cold_token == reused_token,
            "parity": parity,
            "ledger": ledger,
        }
    )
    return CacheReuseBoundaryLedgerReport(
        scenario=f"image_prefix_boundary_ledger_{parity_order}",
        model=model_path,
        image=images,
        top_k=top_k,
        prompt_tokens=plan.prompt_tokens,
        reused_tokens=plan.reused_tokens,
        image_token_id=image_token_id,
        image_token_in_prompt=(
            image_token_id is not None and image_token_id in first_ids
        ),
        image_token_in_suffix=(
            image_token_id is not None and image_token_id in suffix_tokens
        ),
        used_boundary_restore=plan.used_boundary_restore,
        unsafe_rewind_refused=plan.unsafe_rewind_refused,
        generated_tail_removed=plan.generated_tail_removed,
        recoverable=plan.recoverable,
        cold_token=cold_token,
        reused_token=reused_token,
        tokens_match=cold_token == reused_token,
        parity=parity,
        ledger=ledger,
    )


def run_text_prefix_boundary_only_parity(
    *,
    model_path: str,
    image: Optional[Union[str, Sequence[str]]],
    prefix_prompt: str,
    second_suffix: str,
    top_k: int,
    resize_shape: Optional[Union[int, Sequence[int]]] = None,
    trust_remote_code: bool = False,
    processor_kwargs: Optional[Dict[str, Any]] = None,
    parity_order: str = "cold-first",
) -> CacheReuseParityReport:
    model, processor = load(model_path, trust_remote_code=trust_remote_code)
    config = _config_dict(model)
    first_prompt = apply_chat_template(processor, config, prefix_prompt, num_images=0)
    first_inputs = prepare_prompt_inputs(
        model,
        processor,
        first_prompt,
        image=None,
        resize_shape=resize_shape,
        processor_kwargs=processor_kwargs,
    )
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    suffix_tokens = tokenizer.encode(second_suffix, add_special_tokens=False)
    second_inputs = append_suffix_tokens_to_inputs(first_inputs, suffix_tokens)
    second_ids = second_inputs["input_ids"].flatten().tolist()

    initial_state = PromptCacheState()
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    generate(
        model,
        processor,
        "",
        image=None,
        max_tokens=1,
        resize_shape=resize_shape,
        prompt_cache_state=initial_state,
        **generation_kwargs_from_inputs(first_inputs),
    )
    if initial_state.boundary_cache is None or initial_state.boundary_token_ids is None:
        raise RuntimeError("Prompt cache state did not capture a prompt boundary.")

    state = PromptCacheState()
    state.update(initial_state.boundary_token_ids, initial_state.boundary_cache)
    plan = inspect_prompt_reuse(state, second_ids)

    if parity_order == "cold-first":
        cold_result = generate(
            model,
            processor,
            "",
            image=None,
            max_tokens=1,
            resize_shape=resize_shape,
            **generation_kwargs_from_inputs(second_inputs),
        )
        reused_result = generate(
            model,
            processor,
            "",
            image=None,
            max_tokens=1,
            resize_shape=resize_shape,
            prompt_cache_state=state,
            **generation_kwargs_from_inputs(second_inputs),
        )
    elif parity_order == "reused-first":
        reused_result = generate(
            model,
            processor,
            "",
            image=None,
            max_tokens=1,
            resize_shape=resize_shape,
            prompt_cache_state=state,
            **generation_kwargs_from_inputs(second_inputs),
        )
        cold_result = generate(
            model,
            processor,
            "",
            image=None,
            max_tokens=1,
            resize_shape=resize_shape,
            **generation_kwargs_from_inputs(second_inputs),
        )
    else:
        raise ValueError(f"Unsupported parity order: {parity_order}")

    metrics = prompt_cache_state_metrics(state)
    peak_memory_bytes = (
        int(mx.get_peak_memory()) if hasattr(mx, "get_peak_memory") else None
    )
    cold_token = _token_value(cold_result.token)
    reused_token = _token_value(reused_result.token)
    return CacheReuseParityReport(
        scenario=f"text_prefix_boundary_only_parity_{parity_order}",
        model=model_path,
        image=None,
        top_k=top_k,
        prompt_tokens=plan.prompt_tokens,
        reused_tokens=plan.reused_tokens,
        boundary_count=metrics["boundary_count"],
        main_cache_bytes=metrics["main_cache_bytes"],
        boundary_cache_bytes=metrics["boundary_cache_bytes"],
        total_cache_bytes=metrics["total_cache_bytes"],
        peak_memory_bytes=peak_memory_bytes,
        image_token_id=_image_token_id(model),
        image_token_in_prompt=False,
        image_token_in_suffix=False,
        used_boundary_restore=plan.used_boundary_restore,
        unsafe_rewind_refused=plan.unsafe_rewind_refused,
        generated_tail_removed=plan.generated_tail_removed,
        recoverable=plan.recoverable,
        cold_token=cold_token,
        reused_token=reused_token,
        tokens_match=cold_token == reused_token,
        parity=compare_topk_logprobs(
            cold_result.logprobs, reused_result.logprobs, top_k
        ),
    )


def run_image_multiturn_parity(
    *,
    model_path: str,
    image: Union[str, Sequence[str]],
    prefix_prompt: str,
    second_suffix: str,
    top_k: int,
    resize_shape: Optional[Union[int, Sequence[int]]] = None,
    trust_remote_code: bool = False,
    processor_kwargs: Optional[Dict[str, Any]] = None,
    parity_order: str = "cold-first",
) -> CacheReuseParityReport:
    images = _as_list(image)
    model, processor = load(model_path, trust_remote_code=trust_remote_code)
    config = _config_dict(model)
    num_images = len(images or [])
    first_prompt = apply_chat_template(
        processor, config, prefix_prompt, num_images=num_images
    )
    first_inputs = prepare_prompt_inputs(
        model,
        processor,
        first_prompt,
        image=images,
        resize_shape=resize_shape,
        processor_kwargs=processor_kwargs,
    )
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    suffix_tokens = tokenizer.encode(second_suffix, add_special_tokens=False)
    first_ids = first_inputs["input_ids"].flatten().tolist()
    image_token_id = _image_token_id(model)

    state = PromptCacheState()
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    generate(
        model,
        processor,
        "",
        image=None,
        max_tokens=1,
        resize_shape=resize_shape,
        prompt_cache_state=state,
        **generation_kwargs_from_inputs(first_inputs),
    )
    if state.token_ids is None:
        raise RuntimeError("Prompt cache state did not capture the first turn.")

    second_inputs = append_suffix_tokens_to_cached_turn_inputs(
        first_inputs,
        cached_token_ids=state.token_ids,
        suffix_tokens=suffix_tokens,
    )
    second_ids = second_inputs["input_ids"].flatten().tolist()
    plan = inspect_prompt_reuse(state, second_ids)
    if parity_order == "cold-first":
        cold_result = generate(
            model,
            processor,
            "",
            image=None,
            max_tokens=1,
            resize_shape=resize_shape,
            **generation_kwargs_from_inputs(second_inputs),
        )
        reused_result = generate(
            model,
            processor,
            "",
            image=None,
            max_tokens=1,
            resize_shape=resize_shape,
            prompt_cache_state=state,
            **generation_kwargs_from_inputs(second_inputs),
        )
    elif parity_order == "reused-first":
        reused_result = generate(
            model,
            processor,
            "",
            image=None,
            max_tokens=1,
            resize_shape=resize_shape,
            prompt_cache_state=state,
            **generation_kwargs_from_inputs(second_inputs),
        )
        cold_result = generate(
            model,
            processor,
            "",
            image=None,
            max_tokens=1,
            resize_shape=resize_shape,
            **generation_kwargs_from_inputs(second_inputs),
        )
    else:
        raise ValueError(f"Unsupported parity order: {parity_order}")

    metrics = prompt_cache_state_metrics(state)
    peak_memory_bytes = (
        int(mx.get_peak_memory()) if hasattr(mx, "get_peak_memory") else None
    )
    cold_token = _token_value(cold_result.token)
    reused_token = _token_value(reused_result.token)
    return CacheReuseParityReport(
        scenario=f"image_multiturn_text_followup_parity_{parity_order}",
        model=model_path,
        image=images,
        top_k=top_k,
        prompt_tokens=plan.prompt_tokens,
        reused_tokens=plan.reused_tokens,
        boundary_count=metrics["boundary_count"],
        main_cache_bytes=metrics["main_cache_bytes"],
        boundary_cache_bytes=metrics["boundary_cache_bytes"],
        total_cache_bytes=metrics["total_cache_bytes"],
        peak_memory_bytes=peak_memory_bytes,
        image_token_id=image_token_id,
        image_token_in_prompt=(
            image_token_id is not None and image_token_id in first_ids
        ),
        image_token_in_suffix=(
            image_token_id is not None and image_token_id in suffix_tokens
        ),
        used_boundary_restore=plan.used_boundary_restore,
        unsafe_rewind_refused=plan.unsafe_rewind_refused,
        generated_tail_removed=plan.generated_tail_removed,
        recoverable=plan.recoverable,
        cold_token=cold_token,
        reused_token=reused_token,
        tokens_match=cold_token == reused_token,
        parity=compare_topk_logprobs(
            cold_result.logprobs, reused_result.logprobs, top_k
        ),
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a structured VLM prompt-cache reuse smoke."
    )
    parser.add_argument("--model", required=True, help="Local path or HF repo id.")
    parser.add_argument(
        "--scenario",
        choices=[
            "image-prefix",
            "image-prefix-batched",
            "image-multiturn",
            "image-prefix-parity",
            "image-prefix-boundary-only-parity",
            "image-prefix-boundary-ledger",
            "image-multiturn-parity",
            "text-prefix-boundary-only-parity",
        ],
        default="image-prefix",
    )
    parser.add_argument(
        "--image", nargs="+", default=None, help="Image path(s) or URL(s)."
    )
    parser.add_argument(
        "--prefix-prompt",
        default="Describe this image briefly.",
        help="Prompt shared by both turns.",
    )
    parser.add_argument(
        "--second-suffix",
        default="\nNow answer with one short additional detail.",
        help="Suffix appended to the second prompt after the shared prefix.",
    )
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--parity-order",
        choices=["cold-first", "reused-first"],
        default="cold-first",
    )
    parser.add_argument("--resize-shape", type=int, nargs="+", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--processor-kwargs", type=json.loads, default={})
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    scenario_runners = {
        "image-prefix": run_image_prefix_smoke,
        "image-prefix-batched": run_image_prefix_batched_smoke,
        "image-multiturn": run_image_multiturn_smoke,
        "image-prefix-parity": run_image_prefix_parity,
        "image-prefix-boundary-only-parity": run_image_prefix_boundary_only_parity,
        "image-prefix-boundary-ledger": run_image_prefix_boundary_ledger,
        "image-multiturn-parity": run_image_multiturn_parity,
        "text-prefix-boundary-only-parity": run_text_prefix_boundary_only_parity,
    }
    run_smoke = scenario_runners[args.scenario]
    if args.scenario.startswith("image-") and args.image is None:
        raise SystemExit(f"--image is required for scenario {args.scenario}")
    common_kwargs = dict(
        model_path=args.model,
        image=args.image,
        prefix_prompt=args.prefix_prompt,
        second_suffix=args.second_suffix,
        resize_shape=args.resize_shape,
        trust_remote_code=args.trust_remote_code,
        processor_kwargs=args.processor_kwargs,
    )
    if args.scenario.endswith("-parity") or args.scenario.endswith("-ledger"):
        report = run_smoke(
            top_k=args.top_k, parity_order=args.parity_order, **common_kwargs
        )
    else:
        report = run_smoke(max_tokens=args.max_tokens, **common_kwargs)
    payload = json.dumps(report.to_dict(), indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
