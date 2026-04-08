from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _is_gemini_endpoint(api_base_url: str) -> bool:
    return "generativelanguage.googleapis.com" in api_base_url.lower()


def _is_hf_router_endpoint(api_base_url: str) -> bool:
    lowered = api_base_url.lower()
    return "router.huggingface.co" in lowered or "api-inference.huggingface.co" in lowered


def _resolve_api_key(api_base_url: str, explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    hf_key = os.getenv("HF_TOKEN")
    if _is_gemini_endpoint(api_base_url):
        return gemini_key or openai_key or hf_key
    if _is_hf_router_endpoint(api_base_url):
        return hf_key or openai_key or gemini_key
    return openai_key or hf_key or gemini_key


class LLMJudge:
    """Cached audience-fit and coherence judge with deterministic fallback."""

    def __init__(
        self,
        cache_path: str | Path | None = None,
        api_base_url: str | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
    ) -> None:
        self.cache_path = Path(cache_path or Path(__file__).with_name(".llm_judge_cache.json"))
        self.api_base_url = api_base_url or os.getenv(
            "API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        self.api_key = _resolve_api_key(self.api_base_url, explicit=api_key)
        self.model_name = model_name or os.getenv("MODEL_NAME", "gemini-2.0-flash")
        self._cache = self._load_cache()

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        if not self.cache_path.exists():
            return {}
        try:
            data = json.loads(self.cache_path.read_text())
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_cache(self) -> None:
        try:
            self.cache_path.write_text(json.dumps(self._cache, indent=2, sort_keys=True))
        except OSError:
            pass

    def _cache_key(
        self,
        *,
        audience: str,
        reading_level: str,
        tone: str,
        message: str,
        required_elements: list[str],
        forbidden_phrases: list[str],
    ) -> str:
        payload = {
            "audience": audience,
            "reading_level": reading_level,
            "tone": tone,
            "message": _normalize_text(message),
            "required_elements": required_elements,
            "forbidden_phrases": forbidden_phrases,
            "model_name": self.model_name,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def judge_message(
        self,
        *,
        audience: str,
        reading_level: str,
        tone: str,
        message: str,
        required_elements: list[str],
        forbidden_phrases: list[str],
    ) -> dict[str, Any]:
        key = self._cache_key(
            audience=audience,
            reading_level=reading_level,
            tone=tone,
            message=message,
            required_elements=required_elements,
            forbidden_phrases=forbidden_phrases,
        )
        if key in self._cache:
            return self._cache[key]

        result = self._judge_with_api(
            audience=audience,
            reading_level=reading_level,
            tone=tone,
            message=message,
            required_elements=required_elements,
            forbidden_phrases=forbidden_phrases,
        )
        if result is None:
            result = self._heuristic_judge(
                audience=audience,
                reading_level=reading_level,
                tone=tone,
                message=message,
                required_elements=required_elements,
                forbidden_phrases=forbidden_phrases,
            )
        self._cache[key] = result
        self._save_cache()
        return result

    def _judge_with_api(
        self,
        *,
        audience: str,
        reading_level: str,
        tone: str,
        message: str,
        required_elements: list[str],
        forbidden_phrases: list[str],
    ) -> dict[str, Any] | None:
        if not self.api_key:
            return None

        try:
            client = OpenAI(base_url=self.api_base_url, api_key=self.api_key, timeout=20.0)
            prompt = {
                "audience": audience,
                "reading_level": reading_level,
                "tone": tone,
                "required_elements": required_elements,
                "forbidden_phrases": forbidden_phrases,
                "message": message,
            }
            completion = client.chat.completions.create(
                model=self.model_name,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Return strict JSON with keys: audience_fit, coherence, keyword_stuffing, "
                            "hedging, notes. Scores must be floats from 0 to 1. Keep notes short."
                        ),
                    },
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=True)},
                ],
            )
            content = completion.choices[0].message.content or ""
            parsed = json.loads(content)
            return {
                "audience_fit": _clamp(float(parsed.get("audience_fit", 0.5))),
                "coherence": _clamp(float(parsed.get("coherence", 0.5))),
                "keyword_stuffing": bool(parsed.get("keyword_stuffing", False)),
                "hedging": bool(parsed.get("hedging", False)),
                "notes": [str(note) for note in parsed.get("notes", [])[:3]],
                "source": "api",
            }
        except Exception:
            return None

    def _heuristic_judge(
        self,
        *,
        audience: str,
        reading_level: str,
        tone: str,
        message: str,
        required_elements: list[str],
        forbidden_phrases: list[str],
    ) -> dict[str, Any]:
        normalized = _normalize_text(message)
        tokens = re.findall(r"[a-z0-9']+", normalized)
        unique_ratio = len(set(tokens)) / max(len(tokens), 1)
        sentences = [part.strip() for part in re.split(r"[.!?]+", normalized) if part.strip()]
        avg_sentence_len = len(tokens) / max(len(sentences), 1)
        repeated_phrases = len(re.findall(r"\b(\w+)(?:\s+\1){2,}\b", normalized))
        forbidden_hits = sum(1 for phrase in forbidden_phrases if phrase.lower() in normalized)
        hedge_hits = sum(
            1
            for phrase in (
                "we don't know",
                "we do not know",
                "cannot confirm",
                "can't confirm",
                "no comment",
                "ongoing investigation",
            )
            if phrase in normalized
        )
        top_count = max((tokens.count(token) for token in set(tokens)), default=0)
        stuffing = bool(tokens and top_count / max(len(tokens), 1) > 0.18)

        coherence = 0.45
        coherence += 0.15 if unique_ratio >= 0.55 else 0.0
        coherence += 0.1 if 6 <= avg_sentence_len <= 28 else -0.05
        coherence += 0.1 if len(tokens) >= 20 else -0.05
        coherence -= 0.2 if repeated_phrases else 0.0
        coherence -= min(forbidden_hits * 0.08, 0.24)
        coherence -= min(hedge_hits * 0.1, 0.3)
        coherence -= 0.15 if stuffing else 0.0

        audience_fit = 0.55
        if reading_level == "simple":
            audience_fit += 0.15 if avg_sentence_len <= 18 else -0.1
        elif reading_level == "professional":
            audience_fit += 0.1 if 10 <= avg_sentence_len <= 24 else -0.05
        elif reading_level == "legal":
            audience_fit += 0.15 if any(word in normalized for word in ("article", "regulator", "notify", "investigation", "acknowledged")) else 0.0

        if tone == "empathetic" and any(word in normalized for word in ("sorry", "understand", "support", "apologize")):
            audience_fit += 0.05
        if tone == "formal" and any(word in normalized for word in ("hereby", "pursuant", "acknowledge", "formal")):
            audience_fit += 0.05
        if tone == "professional" and any(word in normalized for word in ("confirmed", "statement", "response", "update")):
            audience_fit += 0.05

        notes: list[str] = []
        if stuffing:
            notes.append("Message looks repetitive and may be keyword stuffed.")
        if hedge_hits >= 2:
            notes.append("Message leans on hedging instead of concrete guidance.")
        if forbidden_hits:
            notes.append("Message includes audience-risky language.")
        if not notes:
            notes.append("Heuristic fallback used; no live judge call.")

        return {
            "audience_fit": _clamp(audience_fit),
            "coherence": _clamp(coherence),
            "keyword_stuffing": stuffing,
            "hedging": hedge_hits >= 2,
            "notes": notes,
            "source": "heuristic",
        }
