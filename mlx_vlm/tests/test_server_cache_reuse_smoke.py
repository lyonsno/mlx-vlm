from __future__ import annotations

from mlx_vlm.server_cache_reuse_smoke import (
    ServerCacheReuseSmokeConfig,
    build_chat_payload,
    classify_receipt,
    run_server_cache_reuse_smoke,
)


class FakeHTTPClient:
    def __init__(self):
        self.posts = []
        self.gets = []
        self._stats = [
            {"enabled": True, "lookups_hit": 0, "matched_tokens": 0, "stores": 0},
            {"enabled": True, "lookups_hit": 0, "matched_tokens": 0, "stores": 2},
            {"enabled": True, "lookups_hit": 1, "matched_tokens": 48, "stores": 2},
        ]
        self._metrics = {
            "latest": {
                "endpoint": "/chat/completions",
                "backend": "continuous_batching",
                "image_count": 1,
                "apc_enabled": True,
                "prompt_tokens": 96,
                "completion_tokens": 2,
            },
            "recent": [
                {"image_count": 1, "apc_enabled": True, "prompt_tokens": 96},
                {"image_count": 1, "apc_enabled": True, "prompt_tokens": 99},
            ],
            "server": {
                "apc": {
                    "enabled": True,
                    "lookups_hit": 1,
                    "matched_tokens": 48,
                    "stores": 2,
                }
            },
        }

    def post_json(self, path, payload=None, headers=None):
        self.posts.append((path, payload, headers))
        if path == "/v1/cache/reset":
            return {"enabled": True, "status": "cleared"}
        return {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {
                "prompt_tokens": 96,
                "completion_tokens": 2,
                "total_tokens": 98,
            },
        }

    def get_json(self, path):
        self.gets.append(path)
        if path == "/v1/cache/stats":
            return self._stats.pop(0)
        if path == "/v1/metrics":
            return self._metrics
        raise AssertionError(path)


def test_build_chat_payload_uses_realistic_user_role_image_request():
    payload = build_chat_payload(
        model="demo-model",
        image_url="file:///tmp/counter.png",
        prefix="Shared receipt prefix. " * 4,
        question="What is visible?",
        max_tokens=4,
    )

    assert payload["model"] == "demo-model"
    assert payload["max_tokens"] == 4
    message = payload["messages"][0]
    assert message["role"] == "user"
    assert message["content"][0]["type"] == "text"
    assert message["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "file:///tmp/counter.png"},
    }


def test_classify_receipt_reports_apc_hit_and_image_path():
    report = classify_receipt(
        before_stats={"enabled": True, "lookups_hit": 0, "matched_tokens": 0},
        cold_stats={"enabled": True, "lookups_hit": 0, "matched_tokens": 0},
        warm_stats={"enabled": True, "lookups_hit": 1, "matched_tokens": 48},
        metrics={
            "latest": {"image_count": 1, "apc_enabled": True},
            "recent": [{"image_count": 1}, {"image_count": 1}],
        },
    )

    assert report["apc_reuse_observed"] is True
    assert report["image_payload_observed_by_server"] is True
    assert report["missing_receipt_gap"] is None
    assert report["apc_delta"]["lookups_hit"] == 1
    assert report["apc_delta"]["matched_tokens"] == 48


def test_classify_receipt_names_missing_gap_without_overclaiming():
    report = classify_receipt(
        before_stats={"enabled": True, "lookups_hit": 0, "matched_tokens": 0},
        cold_stats={"enabled": True, "lookups_hit": 0, "matched_tokens": 0},
        warm_stats={"enabled": True, "lookups_hit": 0, "matched_tokens": 0},
        metrics={
            "latest": {"image_count": 1, "apc_enabled": True},
            "recent": [{"image_count": 1}, {"image_count": 1}],
        },
    )

    assert report["apc_reuse_observed"] is False
    assert report["missing_receipt_gap"] == (
        "APC was enabled and image-bearing requests reached server metrics, "
        "but warm request did not increase lookups_hit or matched_tokens."
    )


def test_run_server_cache_reuse_smoke_drives_first_party_endpoints():
    client = FakeHTTPClient()
    config = ServerCacheReuseSmokeConfig(
        base_url="http://server.test",
        model="demo-model",
        image_url="file:///tmp/counter.png",
        prefix_repetitions=8,
        max_tokens=2,
        tenant="receipt-lane",
    )

    report = run_server_cache_reuse_smoke(config, client=client)

    chat_posts = [post for post in client.posts if post[0] == "/v1/chat/completions"]
    assert len(chat_posts) == 2
    assert chat_posts[0][2] == {"X-APC-Tenant": "receipt-lane"}
    assert chat_posts[0][1]["messages"][0]["role"] == "user"
    assert chat_posts[0][1]["messages"][0]["content"][1]["type"] == "image_url"
    assert client.gets == [
        "/v1/cache/stats",
        "/v1/cache/stats",
        "/v1/cache/stats",
        "/v1/metrics",
    ]
    assert report["receipt"]["apc_reuse_observed"] is True
    assert report["receipt"]["image_payload_observed_by_server"] is True
