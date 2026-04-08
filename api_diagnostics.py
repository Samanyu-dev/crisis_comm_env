from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import httpx


DEFAULT_API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai/",
)
DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash")


def _is_gemini_endpoint(api_base_url: str) -> bool:
    return "generativelanguage.googleapis.com" in api_base_url.lower()


def _is_hf_router_endpoint(api_base_url: str) -> bool:
    lowered = api_base_url.lower()
    return "router.huggingface.co" in lowered or "api-inference.huggingface.co" in lowered


def resolve_api_key(api_base_url: str, explicit: str | None = None) -> tuple[str | None, str]:
    if explicit:
        return explicit, "explicit"
    hf_key = os.getenv("HF_TOKEN")
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    if _is_gemini_endpoint(api_base_url):
        if gemini_key:
            return gemini_key, "GEMINI_API_KEY"
        if openai_key:
            return openai_key, "OPENAI_API_KEY"
        if hf_key:
            return hf_key, "HF_TOKEN"
        return None, "none"
    if _is_hf_router_endpoint(api_base_url):
        if hf_key:
            return hf_key, "HF_TOKEN"
        if openai_key:
            return openai_key, "OPENAI_API_KEY"
        if gemini_key:
            return gemini_key, "GEMINI_API_KEY"
        return None, "none"
    if openai_key:
        return openai_key, "OPENAI_API_KEY"
    if hf_key:
        return hf_key, "HF_TOKEN"
    if gemini_key:
        return gemini_key, "GEMINI_API_KEY"
    return None, "none"


def mask_token(token: str | None) -> str:
    if not token:
        return "missing"
    if len(token) <= 8:
        return "*" * len(token)
    return f"{token[:4]}...{token[-4:]}"


def extract_debug_headers(headers: httpx.Headers) -> dict[str, str]:
    keys = []
    for key in headers.keys():
        lowered = key.lower()
        if "ratelimit" in lowered or lowered in {"retry-after", "x-request-id", "request-id"}:
            keys.append(key)
    return {key: headers.get(key, "") for key in sorted(set(keys), key=str.lower)}


def request_with_metrics(
    client: httpx.Client,
    method: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    response = client.request(method, path, json=payload)
    elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
    return {
        "path": path,
        "status_code": response.status_code,
        "latency_ms": elapsed_ms,
        "rate_headers": extract_debug_headers(response.headers),
        "body_preview": response.text[:240],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe OpenAI-compatible API status, latency, and rate-limit headers.")
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--skip-chat", action="store_true")
    args = parser.parse_args()

    api_key, source = resolve_api_key(args.api_base_url, explicit=args.api_key)
    summary: dict[str, Any] = {
        "api_base_url": args.api_base_url,
        "model": args.model,
        "api_key_source": source,
        "api_key_masked": mask_token(api_key),
        "checks": [],
    }

    if not api_key:
        summary["error"] = "No API key found. Set GEMINI_API_KEY, HF_TOKEN, OPENAI_API_KEY, or --api-key."
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 1

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    with httpx.Client(base_url=args.api_base_url.rstrip("/"), headers=headers, timeout=45.0) as client:
        try:
            summary["checks"].append(request_with_metrics(client, "GET", "/models"))
        except Exception as exc:
            summary["checks"].append({"path": "/models", "error": str(exc)})

        if not args.skip_chat:
            chat_payload = {
                "model": args.model,
                "temperature": 0,
                "max_tokens": 5,
                "messages": [{"role": "user", "content": "Respond with exactly: ok"}],
            }
            try:
                summary["checks"].append(
                    request_with_metrics(client, "POST", "/chat/completions", payload=chat_payload)
                )
            except Exception as exc:
                summary["checks"].append({"path": "/chat/completions", "error": str(exc)})

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
