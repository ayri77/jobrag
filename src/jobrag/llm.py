from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Literal
import requests
import json

from jobrag.settings import SETTINGS

TrimMode = Literal["none", "markers"]

@dataclass(frozen=True)
class OllamaClient:
    host: str = SETTINGS.OLLAMA_HOST
    model: str = SETTINGS.OLLAMA_MODEL
    timeout_s: int = SETTINGS.OLLAMA_TIMEOUT_S
    trim_mode: TrimMode = "markers"

    def _trim(self, text: str) -> str:
        if self.trim_mode != "markers":
            return text
        for marker in ["END_OF_ANSWER", "<|endoftext|>", "<|im_start|>"]:
            if marker in text:
                text = text.split(marker, 1)[0]
        return text.strip()

    def generate(
            self,
            prompt: str,
            system: Optional[str] = None,
            temperature: Optional[float] = 0.0,
    ) -> Dict[str, Any]:

        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": temperature,
                #"stop": ["END_OF_ANSWER"],
            },
        }
        if system:
            payload["system"] = system

        r = requests.post(f"{self.host}/api/generate", json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        data["response"] = self._trim(data.get("response", ""))
        return data

    def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_chars: int = 4000,
    ) -> Dict[str, Any]:

        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": True,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system

        with requests.post(
            f"{self.host}/api/generate",
            json=payload,
            timeout=self.timeout_s,
            stream=True,
        ) as r:
            r.raise_for_status()

            parts: list[str] = []
            last_evt: Dict[str, Any] = {}

            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                evt = json.loads(line)
                last_evt = evt

                chunk = evt.get("response", "")
                if chunk:
                    parts.append(chunk)
                    if len("".join(parts)) >= max_chars:
                        break

                if evt.get("done"):
                    break

        full = self._trim("".join(parts))
        last_evt["response"] = full
        return last_evt