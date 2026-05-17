from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import requests


DEFAULT_PREFIX_SENTENCE = (
    "Cash Register Proof Heist shared image-prefix receipt context: "
    "the same user-role image and long textual prefix should be reused by "
    "the first-party mlx-vlm server APC path before answering the short suffix. "
)


@dataclass
class ServerCacheReuseSmokeConfig:
    base_url: str
    model: str
    image_url: str
    prefix: Optional[str] = None
    prefix_repetitions: int = 64
    max_tokens: int = 8
    tenant: Optional[str] = "cash-register-proof-heist"
    reset_cache: bool = True
    cold_question: str = "Describe the visible scene in one short sentence."
    warm_question: Optional[str] = None
    timeout: float = 600.0


class RequestsHTTPClient:
    def __init__(self, base_url: str, timeout: float):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def post_json(
        self,
        path: str,
        payload: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> dict:
        response = requests.post(
            f"{self.base_url}{path}",
            json=payload,
            headers=dict(headers or {}),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_json(self, path: str) -> dict:
        response = requests.get(f"{self.base_url}{path}", timeout=self.timeout)
        response.raise_for_status()
        return response.json()


def _shared_prefix(config: ServerCacheReuseSmokeConfig) -> str:
    if config.prefix is not None:
        return config.prefix
    repetitions = max(1, int(config.prefix_repetitions))
    return DEFAULT_PREFIX_SENTENCE * repetitions


def _warm_question(config: ServerCacheReuseSmokeConfig) -> str:
    return config.warm_question or config.cold_question


def build_chat_payload(
    *,
    model: str,
    image_url: str,
    prefix: str,
    question: str,
    max_tokens: int,
) -> dict:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{prefix}\n\n{question}"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "max_tokens": int(max_tokens),
        "temperature": 0.0,
        "stream": False,
    }


def _int_value(payload: Mapping[str, Any], key: str) -> int:
    try:
        return int(payload.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def _delta(after: Mapping[str, Any], before: Mapping[str, Any], key: str) -> int:
    return _int_value(after, key) - _int_value(before, key)


def _recent_image_count(metrics: Mapping[str, Any]) -> int:
    recent = metrics.get("recent") or []
    if not isinstance(recent, list):
        return 0
    count = 0
    for item in recent[-2:]:
        if isinstance(item, Mapping) and _int_value(item, "image_count") > 0:
            count += 1
    return count


def classify_receipt(
    *,
    before_stats: Mapping[str, Any],
    cold_stats: Mapping[str, Any],
    warm_stats: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict:
    latest = metrics.get("latest") or {}
    server = metrics.get("server") or {}
    server_apc = server.get("apc") or {}
    apc_enabled = bool(
        warm_stats.get("enabled")
        or latest.get("apc_enabled")
        or server_apc.get("enabled")
    )
    image_payload_observed = (
        _int_value(latest, "image_count") > 0 or _recent_image_count(metrics) >= 2
    )
    apc_delta = {
        "lookups_hit": _delta(warm_stats, before_stats, "lookups_hit"),
        "matched_tokens": _delta(warm_stats, before_stats, "matched_tokens"),
        "exact_hits": _delta(warm_stats, before_stats, "exact_hits"),
        "disk_hits": _delta(warm_stats, before_stats, "disk_hits"),
        "stores": _delta(warm_stats, before_stats, "stores"),
        "served_tokens": _delta(warm_stats, before_stats, "served_tokens"),
    }
    warm_delta = {
        "lookups_hit": _delta(warm_stats, cold_stats, "lookups_hit"),
        "matched_tokens": _delta(warm_stats, cold_stats, "matched_tokens"),
        "exact_hits": _delta(warm_stats, cold_stats, "exact_hits"),
        "disk_hits": _delta(warm_stats, cold_stats, "disk_hits"),
    }
    apc_reuse_observed = apc_enabled and (
        warm_delta["lookups_hit"] > 0
        or warm_delta["matched_tokens"] > 0
        or warm_delta["exact_hits"] > 0
        or warm_delta["disk_hits"] > 0
    )

    missing_gap = None
    if not apc_enabled:
        missing_gap = "APC stats endpoint reported enabled=false for the server run."
    elif not image_payload_observed:
        missing_gap = (
            "Server metrics did not record image_count > 0 for the image-bearing "
            "chat completion requests."
        )
    elif not apc_reuse_observed:
        missing_gap = (
            "APC was enabled and image-bearing requests reached server metrics, "
            "but warm request did not increase lookups_hit or matched_tokens."
        )

    return {
        "apc_enabled": apc_enabled,
        "image_payload_observed_by_server": image_payload_observed,
        "apc_reuse_observed": apc_reuse_observed,
        "apc_delta": apc_delta,
        "warm_request_delta": warm_delta,
        "missing_receipt_gap": missing_gap,
    }


def _usage(response: Mapping[str, Any]) -> dict:
    usage = response.get("usage") or {}
    return {
        "prompt_tokens": _int_value(usage, "prompt_tokens"),
        "completion_tokens": _int_value(usage, "completion_tokens"),
        "total_tokens": _int_value(usage, "total_tokens"),
    }


def _response_text(response: Mapping[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices or not isinstance(choices[0], Mapping):
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    return content if isinstance(content, str) else ""


def run_server_cache_reuse_smoke(
    config: ServerCacheReuseSmokeConfig,
    *,
    client: Optional[Any] = None,
) -> dict:
    client = client or RequestsHTTPClient(config.base_url, config.timeout)
    headers = {"X-APC-Tenant": config.tenant} if config.tenant else {}
    prefix = _shared_prefix(config)

    if config.reset_cache:
        reset_response = client.post_json("/v1/cache/reset", headers=headers)
    else:
        reset_response = None

    before_stats = client.get_json("/v1/cache/stats")
    cold_payload = build_chat_payload(
        model=config.model,
        image_url=config.image_url,
        prefix=prefix,
        question=config.cold_question,
        max_tokens=config.max_tokens,
    )
    warm_payload = build_chat_payload(
        model=config.model,
        image_url=config.image_url,
        prefix=prefix,
        question=_warm_question(config),
        max_tokens=config.max_tokens,
    )

    cold_response = client.post_json(
        "/v1/chat/completions", cold_payload, headers=headers
    )
    cold_stats = client.get_json("/v1/cache/stats")
    warm_response = client.post_json(
        "/v1/chat/completions", warm_payload, headers=headers
    )
    warm_stats = client.get_json("/v1/cache/stats")
    metrics = client.get_json("/v1/metrics")
    receipt = classify_receipt(
        before_stats=before_stats,
        cold_stats=cold_stats,
        warm_stats=warm_stats,
        metrics=metrics,
    )

    return {
        "scenario": "first_party_vlm_server_image_apc_receipt",
        "server": {
            "base_url": config.base_url.rstrip("/"),
            "chat_endpoint": "/v1/chat/completions",
            "cache_stats_endpoint": "/v1/cache/stats",
            "metrics_endpoint": "/v1/metrics",
        },
        "model": config.model,
        "tenant": config.tenant,
        "request_shape": {
            "role": "user",
            "content_types": ["text", "image_url"],
            "image_url": config.image_url,
            "shared_prefix_chars": len(prefix),
            "max_tokens": int(config.max_tokens),
        },
        "reset_response": reset_response,
        "cold_response": {
            "usage": _usage(cold_response),
            "text_preview": _response_text(cold_response)[:200],
        },
        "warm_response": {
            "usage": _usage(warm_response),
            "text_preview": _response_text(warm_response)[:200],
        },
        "apc_stats": {
            "before": dict(before_stats),
            "after_cold": dict(cold_stats),
            "after_warm": dict(warm_stats),
        },
        "metrics": {
            "latest": metrics.get("latest"),
            "recent": metrics.get("recent"),
            "server_apc": (metrics.get("server") or {}).get("apc"),
        },
        "receipt": receipt,
        "claim_boundary": (
            "This smoke proves only first-party server request routing and APC "
            "receipt visibility for the observed run. It does not claim universal "
            "image-prefix distribution equality."
        ),
    }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Send cold/warm image-bearing chat completions to an mlx-vlm server "
            "and emit APC receipt evidence."
        )
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image-url", required=True)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--prefix-repetitions", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--tenant", default="cash-register-proof-heist")
    parser.add_argument(
        "--cold-question",
        default="Describe the visible scene in one short sentence.",
        help="Question used for the cold request.",
    )
    parser.add_argument(
        "--warm-question",
        default=None,
        help=(
            "Question used for the warm request. Defaults to the cold question "
            "so exact APC modes can prove an identical repeated request."
        ),
    )
    parser.add_argument("--no-reset-cache", action="store_true")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    config = ServerCacheReuseSmokeConfig(
        base_url=args.base_url,
        model=args.model,
        image_url=args.image_url,
        prefix=args.prefix,
        prefix_repetitions=args.prefix_repetitions,
        max_tokens=args.max_tokens,
        tenant=args.tenant,
        reset_cache=not args.no_reset_cache,
        cold_question=args.cold_question,
        warm_question=args.warm_question,
        timeout=args.timeout,
    )
    report = run_server_cache_reuse_smoke(config)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        print(encoded)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    return 0 if report["receipt"]["missing_receipt_gap"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
