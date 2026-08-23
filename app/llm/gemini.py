from __future__ import annotations

import json
import socket
import time
from typing import Any, Protocol
from urllib import error, request


class GeminiError(RuntimeError):
    pass


class GeminiUnavailableError(GeminiError):
    pass


class StructuredLLM(Protocol):
    def generate(self, system_prompt: str, prompt: str, response_schema: dict[str, Any]) -> dict[str, Any]: ...


class GeminiProvider:
    """Minimal Gemini REST client with bounded retries for transient failures."""

    transient_statuses = {408, 429, 500, 502, 503, 504}

    def __init__(self, api_key: str | None, model: str, timeout_seconds: float = 10, max_retries: int = 2) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    def generate(self, system_prompt: str, prompt: str, response_schema: dict[str, Any]) -> dict[str, Any]:
        if not self._api_key:
            raise GeminiUnavailableError("GEMINI_API_KEY is not configured.")
        body = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
                "responseJsonSchema": response_schema,
            },
        }
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self._model}:generateContent"
        for attempt in range(self._max_retries + 1):
            try:
                http_request = request.Request(
                    endpoint,
                    data=json.dumps(body).encode("utf-8"),
                    headers={"Content-Type": "application/json", "x-goog-api-key": self._api_key},
                    method="POST",
                )
                with request.urlopen(http_request, timeout=self._timeout_seconds) as response:
                    data = json.loads(response.read().decode("utf-8"))
                return self._parse_response(data)
            except error.HTTPError as exc:
                if exc.code not in self.transient_statuses or attempt == self._max_retries:
                    raise GeminiError(f"Gemini HTTP error: {exc.code}") from exc
            except (error.URLError, TimeoutError, socket.timeout) as exc:
                if attempt == self._max_retries:
                    raise GeminiUnavailableError("Gemini request timed out or was unavailable.") from exc
            if attempt < self._max_retries:
                time.sleep(0.25 * (2**attempt))
        raise GeminiUnavailableError("Gemini did not return a result.")

    @staticmethod
    def _parse_response(data: dict[str, Any]) -> dict[str, Any]:
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            parsed = json.loads(text)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise GeminiError("Gemini returned invalid structured JSON.") from exc
        if not isinstance(parsed, dict):
            raise GeminiError("Gemini structured response must be a JSON object.")
        return parsed
