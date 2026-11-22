from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from google import genai
from google.genai import types

from .cancel import JobCancelledError, check_cancelled, get_current_cancel_token

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

DEFAULT_TEXT_FALLBACK_MODELS = [
    "qwen3.8-max",
    "qwen4-max",
    "glm-5.3",
    "kimi-k3.1-pro",
    "deepseek-v4-pro",
    "glm-5.2",
    "qwen3.8-flash",
    "gemini:gemini-2.5-flash",
]
DEFAULT_MULTIMODAL_FALLBACK_MODELS = [
    "qwen3.8-omni-flash",
    "qwen3.5-omni-plus",
    "qwen3-vl-plus",
    "gemini:gemini-2.5-flash",
]

# Curated verified model catalog shown in Web UI & CLI
MODEL_CATALOG = {
    "text": [
        {"provider": "dashscope", "model": "qwen3.8-max", "label": "百炼 · Qwen3.8-Max (默认推荐·高速旗舰)"},
        {"provider": "dashscope", "model": "qwen4-max", "label": "百炼 · Qwen4-Max (最新旗舰)"},
        {"provider": "dashscope", "model": "glm-5.3", "label": "百炼 · GLM-5.3 (深度推理)"},
        {"provider": "dashscope", "model": "glm-5.3-prime", "label": "百炼 · GLM-5.3-Prime"},
        {"provider": "dashscope", "model": "glm-5.2", "label": "百炼 · GLM-5.2"},
        {"provider": "dashscope", "model": "kimi-k3.1-pro", "label": "百炼 · Kimi-K3.1-Pro"},
        {"provider": "dashscope", "model": "kimi/kimi-k3.1-pro-highspeed", "label": "百炼 · Kimi-K3.1-Pro Highspeed"},
        {"provider": "dashscope", "model": "deepseek-v4-pro", "label": "百炼 · DeepSeek-V4-Pro"},
        {"provider": "dashscope", "model": "deepseek-v4.1-flash", "label": "百炼 · DeepSeek-V4.1-Flash"},
        {"provider": "dashscope", "model": "MiniMax/MiniMax-M3", "label": "百炼 · MiniMax-M3"},
        {"provider": "dashscope", "model": "qwen3.8-flash", "label": "百炼 · Qwen3.8-Flash (高速低成本)"},
        {"provider": "gemini", "model": "gemini-2.5-flash", "label": "Gemini API · 2.5 Flash"},
        {"provider": "vertex", "model": "gemini-2.5-pro", "label": "Vertex AI · Gemini 2.5 Pro"},
    ],
    "multimodal": [
        {"provider": "dashscope", "model": "qwen3.5-omni-plus", "label": "百炼 · Qwen3.5-Omni-Plus (声画同看·默认)"},
        {"provider": "dashscope", "model": "qwen3.8-omni-flash", "label": "百炼 · Qwen3.8-Omni-Flash (声画极速)"},
        {"provider": "dashscope", "model": "qwen3.5-omni-flash", "label": "百炼 · Qwen3.5-Omni-Flash"},
        {"provider": "dashscope", "model": "qwen3-vl-plus", "label": "百炼 · Qwen3-VL-Plus (视频视觉)"},
        {"provider": "gemini", "model": "gemini-2.5-flash", "label": "Gemini API · 2.5 Flash (声画)"},
        {"provider": "vertex", "model": "gemini-2.5-flash", "label": "Vertex AI · Gemini 2.5 Flash (声画)"},
    ],
}


@dataclass
class LLMCallResult:
    text: str
    parsed: Any
    model: str
    provider: str
    prompt_tokens: int
    output_tokens: int
    elapsed_ms: int


def _load_local_env_vars() -> dict[str, str]:
    merged: dict[str, str] = {}
    candidates = [
        Path.home() / "cc" / ".baoyu-skills" / ".env",
        Path.home() / "cc" / ".env",
        Path.home() / "cc" / "Library" / "Tools" / "autosub" / ".env",
        Path.home() / ".env",
    ]
    for env_path in candidates:
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            val = v.strip().strip('"').strip("'")
            if val:
                merged[k.strip()] = val
    return merged


def _detect_adc_project() -> str | None:
    adc = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    if not adc.exists():
        return None
    try:
        data = json.loads(adc.read_text(encoding="utf-8"))
        return data.get("quota_project_id") or data.get("project_id")
    except Exception:
        return None


def _find_env_key(local_env: dict[str, str], *names: str) -> str | None:
    for n in names:
        val = os.environ.get(n) or local_env.get(n)
        if val:
            return val
    return None


def _extract_json(raw: str) -> Any:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m_obj = re.search(r"\{.*\}", text, flags=re.DOTALL)
        m_arr = re.search(r"\[.*\]", text, flags=re.DOTALL)
        for m in (m_obj, m_arr):
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
        raise


def parse_provider_model(spec: str, default_provider: str) -> tuple[str, str]:
    """Parse 'dashscope:qwen3.8-max' or bare 'qwen3.8-max'."""
    if ":" in spec:
        prov, mod = spec.split(":", 1)
        prov = prov.strip().lower()
        if prov in ("dashscope", "bailian", "vertex", "gemini"):
            return ("dashscope" if prov == "bailian" else prov), mod.strip()
    return default_provider, spec.strip()


class ClipCutLLM:
    """Unified multi-provider LLM client: DashScope (阿里云百炼) default, with independent text & multimodal routing."""

    def __init__(self, models_cfg: dict[str, Any]):
        self.models_cfg = models_cfg
        self.local_env = _load_local_env_vars()
        self.dashscope_key = _find_env_key(self.local_env, "DASHSCOPE_API_KEY", "BAILIAN_API_KEY")
        self.gemini_key = _find_env_key(self.local_env, "GEMINI_API_KEY", "GOOGLE_API_KEY")
        self.vertex_project = (
            models_cfg.get("vertex_project")
            or _detect_adc_project()
            or _find_env_key(self.local_env, "GOOGLE_CLOUD_PROJECT", "VERTEX_PROJECT_ID")
            or ""
        )
        self.vertex_location = (
            models_cfg.get("vertex_location")
            or _find_env_key(self.local_env, "GOOGLE_CLOUD_LOCATION", "VERTEX_LOCATION")
            or "global"
        )
        self._vertex_client: genai.Client | None = None
        self._gemini_client: genai.Client | None = None

    def _get_vertex_client(self) -> genai.Client | None:
        if self._vertex_client is not None:
            return self._vertex_client
        sa_path_str = _find_env_key(self.local_env, "VERTEX_SA_KEY_PATH", "GOOGLE_APPLICATION_CREDENTIALS")
        if sa_path_str:
            sa_path = Path(sa_path_str).expanduser()
            if sa_path.exists():
                try:
                    from google.oauth2 import service_account

                    sa_info = json.loads(sa_path.read_text(encoding="utf-8"))
                    sa_proj = self.models_cfg.get("vertex_project") or sa_info.get("project_id") or self.vertex_project
                    creds = service_account.Credentials.from_service_account_file(
                        str(sa_path),
                        scopes=["https://www.googleapis.com/auth/cloud-platform"],
                    )
                    self._vertex_client = genai.Client(
                        vertexai=True,
                        project=sa_proj,
                        location=self.vertex_location,
                        credentials=creds,
                    )
                    return self._vertex_client
                except Exception:
                    pass

        adc = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
        if not self.vertex_project or not adc.exists():
            return None
        try:
            self._vertex_client = genai.Client(
                vertexai=True,
                project=self.vertex_project,
                location=self.vertex_location,
            )
            return self._vertex_client
        except Exception:
            return None

    def _get_gemini_client(self) -> genai.Client | None:
        if self._gemini_client is not None:
            return self._gemini_client
        if not self.gemini_key:
            return None
        try:
            self._gemini_client = genai.Client(
                api_key=self.gemini_key,
                http_options=types.HttpOptions(api_version="v1beta"),
            )
            return self._gemini_client
        except Exception:
            return None

    def _call_dashscope(
        self,
        *,
        model: str,
        prompt: str,
        system_instruction: str | None,
        temperature: float,
        max_output_tokens: int,
        media_bytes: bytes | None,
        media_mime: str | None,
    ) -> LLMCallResult:
        if not self.dashscope_key:
            raise RuntimeError("未检测到 DASHSCOPE_API_KEY（百炼 API Key）")

        t0 = time.monotonic()
        url = f"{DASHSCOPE_BASE_URL}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.dashscope_key}",
            "Content-Type": "application/json",
        }

        messages: list[dict[str, Any]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction.strip()})

        if media_bytes is not None and media_mime:
            b64 = base64.b64encode(media_bytes).decode("ascii")
            media_part: dict[str, Any]
            if media_mime.startswith("video/"):
                media_part = {"type": "video_url", "video_url": {"url": f"data:{media_mime};base64,{b64}"}}
            else:
                media_part = {"type": "image_url", "image_url": {"url": f"data:{media_mime};base64,{b64}"}}
            messages.append(
                {
                    "role": "user",
                    "content": [
                        media_part,
                        {"type": "text", "text": prompt},
                    ],
                }
            )
        else:
            messages.append({"role": "user", "content": prompt})

        _mn = model.lower()
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": min(max_output_tokens, 16384),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if not any(x in _mn for x in ("o1", "o3", "kimi-k2.5", "kimi-k2.6", "kimi-k3")):
            payload["temperature"] = temperature
        is_omni = "omni" in _mn
        if is_omni:
            payload["modalities"] = ["text"]
        elif media_bytes is None:
            payload["response_format"] = {"type": "json_object"}

        raw_chunks: list[str] = []
        usage: dict[str, Any] = {}
        direct_data: dict[str, Any] | None = None
        client_timeout = httpx.Timeout(connect=15.0, read=120.0, write=60.0, pool=15.0)
        token = get_current_cancel_token()
        check_cancelled()
        with httpx.Client(trust_env=False, timeout=client_timeout) as client:
            if token is not None:
                token.register_client(client)
            try:
                for attempt in range(2):
                    check_cancelled()
                    raw_chunks.clear()
                    usage = {}
                    direct_data = None
                    with client.stream("POST", url, headers=headers, json=payload) as resp:
                        if resp.status_code == 400 and attempt == 0 and "response_format" in payload:
                            # Some reasoning/multimodal models disallow response_format=json_object
                            resp.read()
                            payload.pop("response_format", None)
                            continue
                        if resp.status_code != 200:
                            err_body = resp.read().decode("utf-8", errors="replace")
                            raise RuntimeError(f"HTTP {resp.status_code}: {err_body[:300]}")

                        ctype = (resp.headers.get("content-type") or "").lower()
                        if "application/json" in ctype:
                            direct_data = json.loads(resp.read().decode("utf-8", errors="replace"))
                            break

                        for line in resp.iter_lines():
                            check_cancelled()
                            if not line or not line.startswith("data:"):
                                continue
                            data_str = line[5:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                obj = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue
                            if isinstance(obj.get("usage"), dict):
                                usage = obj["usage"]
                            for ch in obj.get("choices") or []:
                                delta = ch.get("delta") or {}
                                part = delta.get("content")
                                if part:
                                    raw_chunks.append(str(part))
                        break
            except JobCancelledError:
                raise
            except Exception:
                if token is not None and token.is_cancelled:
                    raise JobCancelledError("任务已手动中止") from None
                raise
            finally:
                if token is not None:
                    token.unregister_client(client)

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        if direct_data is not None:
            choice = (direct_data.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            raw_text = str(msg.get("content") or "").strip()
            usage = direct_data.get("usage") or usage
        else:
            raw_text = "".join(raw_chunks).strip()

        if not raw_text:
            raise ValueError(f"Empty response from dashscope:{model}")

        parsed = _extract_json(raw_text)
        return LLMCallResult(
            text=raw_text,
            parsed=parsed,
            model=model,
            provider="dashscope",
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            elapsed_ms=elapsed_ms,
        )

    def _call_google(
        self,
        *,
        provider: str,
        model: str,
        prompt: str,
        system_instruction: str | None,
        temperature: float,
        max_output_tokens: int,
        media_bytes: bytes | None,
        media_mime: str | None,
    ) -> LLMCallResult:
        client = self._get_vertex_client() if provider == "vertex" else self._get_gemini_client()
        if client is None:
            raise RuntimeError(f"{provider} 客户端不可用（缺少凭证或 API Key）")

        t0 = time.monotonic()
        combined_prompt = f"{system_instruction.strip()}\n\n{prompt}" if system_instruction else prompt
        contents: Any
        if media_bytes is not None and media_mime:
            contents = [
                types.Part.from_bytes(data=media_bytes, mime_type=media_mime),
                combined_prompt,
            ]
        else:
            contents = combined_prompt

        eff_max = max(int(max_output_tokens or 0), 8192)
        cfg_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": eff_max,
            "response_mime_type": "application/json",
        }
        if "2.5" in model.lower():
            cfg_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=min(1024, max(128, eff_max // 8))
            )
        cfg = types.GenerateContentConfig(**cfg_kwargs)
        check_cancelled()
        resp = client.models.generate_content(model=model, contents=contents, config=cfg)
        check_cancelled()
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        raw_text = (resp.text or "").strip()
        if not raw_text:
            raise ValueError(f"Empty response from {provider}:{model}")
        try:
            parsed = _extract_json(raw_text)
        except Exception as exc:
            finish_reason = None
            try:
                cands = getattr(resp, "candidates", None) or []
                if cands:
                    finish_reason = getattr(cands[0], "finish_reason", None)
            except Exception:
                pass
            raise ValueError(f"JSON 解析失败 (finish_reason={finish_reason}): {exc}") from exc
        usage = getattr(resp, "usage_metadata", None)
        return LLMCallResult(
            text=raw_text,
            parsed=parsed,
            model=model,
            provider=provider,
            prompt_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
            output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
            elapsed_ms=elapsed_ms,
        )

    def generate_json(
        self,
        *,
        prompt: str,
        model: str,
        provider: str | None = None,
        fallback_models: list[str] | None = None,
        system_instruction: str | None = None,
        temperature: float = 0.25,
        max_output_tokens: int = 16384,
        media_bytes: bytes | None = None,
        media_mime: str | None = None,
    ) -> LLMCallResult:
        default_prov = (provider or ("dashscope" if self.dashscope_key else "gemini")).lower()
        if default_prov == "bailian":
            default_prov = "dashscope"

        attempts: list[tuple[str, str]] = []
        primary = parse_provider_model(model, default_prov)
        attempts.append(primary)
        for fb in fallback_models or []:
            if not fb:
                continue
            pair = parse_provider_model(fb, default_prov)
            if pair not in attempts:
                attempts.append(pair)

        errors: list[str] = []
        skip_dashscope = False
        for prov, mod in attempts:
            check_cancelled()
            if skip_dashscope and prov == "dashscope":
                continue
            try:
                if prov == "dashscope":
                    return self._call_dashscope(
                        model=mod,
                        prompt=prompt,
                        system_instruction=system_instruction,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                        media_bytes=media_bytes,
                        media_mime=media_mime,
                    )
                else:
                    return self._call_google(
                        provider=prov,
                        model=mod,
                        prompt=prompt,
                        system_instruction=system_instruction,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                        media_bytes=media_bytes,
                        media_mime=media_mime,
                    )
            except JobCancelledError:
                raise
            except Exception as exc:
                msg = str(exc)
                if prov == "dashscope" and (
                    "data_inspection_failed" in msg
                    or "inappropriate content" in msg.lower()
                ):
                    skip_dashscope = True
                errors.append(f"{prov}:{mod} -> {exc}")
                continue

        raise RuntimeError("LLM 调用失败（所有配置模型均不可用）:\n" + "\n".join(errors[-5:]))
