import json

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache, make_prompt_cache_boundary

from mlx_vlm.cache_reuse_smoke import (
    CacheReuseSmokeReport,
    PromptReusePlan,
    append_suffix_tokens_to_cached_turn_inputs,
    append_suffix_tokens_to_inputs,
    cache_nbytes,
    compare_topk_logprobs,
    inspect_prompt_reuse,
    parse_args,
    prompt_cache_state_metrics,
    topk_logprob_snapshot,
)
from mlx_vlm.generate import PromptCacheState


def _mixed_cache(prefix_len=2):
    prompt_cache = [ArraysCache(size=1), KVCache()]
    prompt_cache[0][0] = mx.ones((1, 2, 3))
    keys = mx.arange(prefix_len * 4).reshape(1, 1, prefix_len, 4)
    prompt_cache[1].update_and_fetch(keys, keys + 100)
    return prompt_cache


def test_cache_nbytes_recurses_cache_containers():
    prompt_cache = _mixed_cache()
    expected = prompt_cache[0].nbytes + prompt_cache[1].nbytes

    assert cache_nbytes(prompt_cache) == expected
    assert cache_nbytes({"cache": prompt_cache}) == expected
    assert cache_nbytes(None) == 0


def test_prompt_cache_state_metrics_reports_boundary_bytes():
    boundary = _mixed_cache(prefix_len=2)
    final_cache = make_prompt_cache_boundary(boundary)
    tail = mx.ones((1, 1, 1, 4))
    final_cache[1].update_and_fetch(tail, tail + 100)

    state = PromptCacheState()
    state.update(
        [1, 2, 3],
        final_cache,
        boundary_token_ids=[1, 2],
        boundary_cache=boundary,
    )

    metrics = prompt_cache_state_metrics(state)

    assert metrics["boundary_count"] == 1
    assert metrics["boundary_cache_bytes"] == cache_nbytes(boundary)
    assert metrics["main_cache_bytes"] == cache_nbytes(final_cache)
    assert metrics["total_cache_bytes"] == cache_nbytes(boundary) + cache_nbytes(
        final_cache
    )


def test_inspect_prompt_reuse_identifies_boundary_restore_and_refused_rewind():
    boundary = _mixed_cache(prefix_len=2)
    final_cache = make_prompt_cache_boundary(boundary)
    tail = mx.ones((1, 1, 2, 4))
    final_cache[1].update_and_fetch(tail, tail + 100)

    state = PromptCacheState()
    state.update(
        [1, 2, 9, 9],
        final_cache,
        boundary_token_ids=[1, 2],
        boundary_cache=boundary,
    )

    plan = inspect_prompt_reuse(state, [1, 2, 3])

    assert plan == PromptReusePlan(
        prompt_tokens=3,
        reused_tokens=2,
        used_boundary_restore=True,
        unsafe_rewind_refused=True,
        generated_tail_removed=True,
        recoverable=True,
    )


def test_inspect_prompt_reuse_identifies_full_turn_cache_extension():
    boundary = _mixed_cache(prefix_len=2)
    final_cache = make_prompt_cache_boundary(boundary)
    tail = mx.ones((1, 1, 1, 4))
    final_cache[1].update_and_fetch(tail, tail + 100)

    state = PromptCacheState()
    state.update(
        [1, 2, 901],
        final_cache,
        boundary_token_ids=[1, 2],
        boundary_cache=boundary,
    )

    plan = inspect_prompt_reuse(state, [1, 2, 901, 3, 4])

    assert plan == PromptReusePlan(
        prompt_tokens=5,
        reused_tokens=3,
        used_boundary_restore=False,
        unsafe_rewind_refused=False,
        generated_tail_removed=False,
        recoverable=True,
    )


def test_append_suffix_tokens_to_inputs_preserves_exact_prefix():
    inputs = {
        "input_ids": mx.array([[10, 20, 30]], dtype=mx.int32),
        "attention_mask": mx.array([[1, 1, 1]], dtype=mx.int32),
        "pixel_values": mx.ones((1, 3, 2, 2)),
        "image_grid_thw": mx.array([[1, 2, 2]], dtype=mx.int32),
    }

    extended = append_suffix_tokens_to_inputs(inputs, [40, 50])

    assert extended["input_ids"].tolist() == [[10, 20, 30, 40, 50]]
    assert extended["attention_mask"].tolist() == [[1, 1, 1, 1, 1]]
    assert mx.array_equal(extended["pixel_values"], inputs["pixel_values"])
    assert mx.array_equal(extended["image_grid_thw"], inputs["image_grid_thw"])


def test_append_suffix_tokens_to_cached_turn_inputs_extends_generated_state():
    inputs = {
        "input_ids": mx.array([[10, 20, 30]], dtype=mx.int32),
        "attention_mask": mx.array([[1, 1, 1]], dtype=mx.int32),
        "pixel_values": mx.ones((1, 3, 2, 2)),
        "image_grid_thw": mx.array([[1, 2, 2]], dtype=mx.int32),
    }

    extended = append_suffix_tokens_to_cached_turn_inputs(
        inputs,
        cached_token_ids=[10, 20, 30, 901, 902],
        suffix_tokens=[40, 50],
    )

    assert extended["input_ids"].tolist() == [[10, 20, 30, 901, 902, 40, 50]]
    assert extended["attention_mask"].tolist() == [[1, 1, 1, 1, 1, 1, 1]]
    assert mx.array_equal(extended["pixel_values"], inputs["pixel_values"])
    assert mx.array_equal(extended["image_grid_thw"], inputs["image_grid_thw"])


def test_topk_logprob_snapshot_records_ranked_tokens_and_values():
    snapshot = topk_logprob_snapshot(mx.array([-4.0, -1.0, -3.0, -2.0]), k=3)

    assert snapshot == [
        {"token_id": 1, "logprob": -1.0},
        {"token_id": 3, "logprob": -2.0},
        {"token_id": 2, "logprob": -3.0},
    ]


def test_compare_topk_logprobs_reports_shared_token_delta():
    cold = mx.array([-4.0, -1.0, -3.0, -2.0])
    reused = mx.array([-4.0, -1.25, -2.75, -2.0])

    comparison = compare_topk_logprobs(cold, reused, k=3)

    assert comparison["cold_topk_token_ids"] == [1, 3, 2]
    assert comparison["reused_topk_token_ids"] == [1, 3, 2]
    assert comparison["cold_argmax_token_id"] == 1
    assert comparison["reused_argmax_token_id"] == 1
    assert comparison["argmax_token_id_match"] is True
    assert comparison["argmax_abs_logprob_delta"] == 0.25
    assert comparison["topk_token_ids_match"] is True
    assert comparison["shared_topk_token_count"] == 3
    assert comparison["max_abs_logprob_delta"] == 0.25


def test_parse_args_exposes_parity_order():
    args = parse_args(
        [
            "--model",
            "model",
            "--image",
            "image.png",
            "--scenario",
            "image-prefix-boundary-only-parity",
            "--parity-order",
            "reused-first",
        ]
    )

    assert args.scenario == "image-prefix-boundary-only-parity"
    assert args.parity_order == "reused-first"


def test_parse_args_allows_text_boundary_parity_without_image():
    args = parse_args(
        [
            "--model",
            "model",
            "--scenario",
            "text-prefix-boundary-only-parity",
            "--parity-order",
            "reused-first",
        ]
    )

    assert args.scenario == "text-prefix-boundary-only-parity"
    assert args.image is None
    assert args.parity_order == "reused-first"


def test_smoke_report_json_preserves_answer_bank_fields():
    report = CacheReuseSmokeReport(
        scenario="image_prefix_diverged_suffix",
        model="local-model",
        image=["image.png"],
        prompt_tokens=12,
        reused_tokens=10,
        boundary_count=1,
        main_cache_bytes=100,
        boundary_cache_bytes=40,
        total_cache_bytes=140,
        peak_memory_bytes=1024,
        image_token_id=151655,
        image_token_in_prompt=True,
        image_token_in_suffix=False,
        first_wall_time_s=1.25,
        second_wall_time_s=0.75,
        used_boundary_restore=True,
        unsafe_rewind_refused=True,
        generated_tail_removed=True,
        recoverable=True,
        image_state_contract="reprimed_or_not_in_suffix",
        first_text="first",
        second_text="second",
    )

    payload = report.to_dict()
    encoded = json.dumps(payload, sort_keys=True)

    assert json.loads(encoded)["scenario"] == "image_prefix_diverged_suffix"
    assert payload["image_token_in_prompt"] is True
    assert payload["image_token_in_suffix"] is False
    assert payload["used_boundary_restore"] is True
    assert payload["boundary_cache_bytes"] == 40
