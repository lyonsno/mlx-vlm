import argparse
import json
import numbers
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import mlx.core as mx

from .generate import PromptCacheState, generate, normalize_resize_shape
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


def _input_ids_for_prompt(
    model: Any,
    processor: Any,
    prompt: str,
    *,
    image: Optional[List[str]],
    resize_shape: Optional[Union[int, Sequence[int]]],
    processor_kwargs: Optional[Dict[str, Any]] = None,
) -> List[int]:
    inputs = prepare_inputs(
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
    return inputs["input_ids"].flatten().tolist()


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
    second_prompt = apply_chat_template(
        processor, config, prefix_prompt + second_suffix, num_images=num_images
    )

    state = PromptCacheState()
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    first_start = time.perf_counter()
    first_result = generate(
        model,
        processor,
        first_prompt,
        image=images,
        max_tokens=max_tokens,
        resize_shape=resize_shape,
        prompt_cache_state=state,
        **(processor_kwargs or {}),
    )
    first_wall_time = time.perf_counter() - first_start

    second_ids = _input_ids_for_prompt(
        model,
        processor,
        second_prompt,
        image=images,
        resize_shape=resize_shape,
        processor_kwargs=processor_kwargs,
    )
    plan = inspect_prompt_reuse(state, second_ids)

    second_start = time.perf_counter()
    second_result = generate(
        model,
        processor,
        second_prompt,
        image=images,
        max_tokens=max_tokens,
        resize_shape=resize_shape,
        prompt_cache_state=state,
        **(processor_kwargs or {}),
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
        first_wall_time_s=first_wall_time,
        second_wall_time_s=second_wall_time,
        used_boundary_restore=plan.used_boundary_restore,
        unsafe_rewind_refused=plan.unsafe_rewind_refused,
        generated_tail_removed=plan.generated_tail_removed,
        recoverable=plan.recoverable,
        image_state_contract=_image_state_contract(model, second_prompt, plan),
        first_text=first_result.text,
        second_text=second_result.text,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a structured VLM prompt-cache reuse smoke."
    )
    parser.add_argument("--model", required=True, help="Local path or HF repo id.")
    parser.add_argument(
        "--image", nargs="+", required=True, help="Image path(s) or URL(s)."
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
    parser.add_argument("--resize-shape", type=int, nargs="+", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--processor-kwargs", type=json.loads, default={})
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = run_image_prefix_smoke(
        model_path=args.model,
        image=args.image,
        prefix_prompt=args.prefix_prompt,
        second_suffix=args.second_suffix,
        max_tokens=args.max_tokens,
        resize_shape=args.resize_shape,
        trust_remote_code=args.trust_remote_code,
        processor_kwargs=args.processor_kwargs,
    )
    payload = json.dumps(report.to_dict(), indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
