"""Provider-neutral synchronous tool-calling transport. No provider fallback."""
from __future__ import annotations

import copy
import ipaddress
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import httpx
from jsonschema import Draft7Validator

REGISTRY = json.loads((Path(__file__).parent / "providers.json").read_text(encoding="utf-8"))
ALIASES = {"google": "gemini", "openai-compatible": "compatible",
           "lm-studio": "lmstudio", "llama.cpp": "llamacpp", "x-ai": "xai"}
LEVELS = ("auto", "none", "low", "medium", "high", "max")


def _flag(env, name):
    value = env.get(name, "0").strip().lower()
    if value not in ("0", "1", "false", "true", "no", "yes"):
        raise ValueError(f"{name} must be a boolean")
    return value in ("1", "true", "yes")


@dataclass(frozen=True)
class Settings:
    provider: str
    protocol: str
    base_url: str
    model: str
    check_model: str
    key: str = field(repr=False)
    key_env: str = ""
    local: bool = False
    force: str = "named"
    reasoning: str = "off"
    timeout: float = 120
    max_output: int = 0

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env
        name = env.get("SCOUT_PROVIDER", "deepseek").strip().lower()
        name = ALIASES.get(name, name)
        if name not in REGISTRY:
            raise ValueError(f"Unknown SCOUT_PROVIDER: {name}")
        p = REGISTRY[name]
        def legacy(field, default=""):
            return env.get("DEEPSEEK_" + field, default) if name == "deepseek" else default
        base = env.get("SCOUT_BASE_URL", legacy("BASE_URL", p["base_url"])).strip().rstrip("/")
        model = env.get("SCOUT_MODEL", legacy("MODEL", p.get("model", ""))).strip()
        check = env.get("SCOUT_CHECK_MODEL", legacy("CHECK_MODEL", p.get("check_model", model))).strip()
        key_env = p["key_env"]
        key = env.get("SCOUT_API_KEY", env.get(key_env, "") if key_env else "").strip()
        local = p["local"] or _flag(env, "SCOUT_LOCAL_ONLY")
        if base:
            u = urlsplit(base)
            if u.scheme not in ("http", "https") or not u.hostname or u.username or u.password or u.query or u.fragment:
                raise ValueError("SCOUT_BASE_URL must be an HTTP(S) base URL without credentials, query or fragment")
            try:
                loopback = ipaddress.ip_address(u.hostname).is_loopback
            except ValueError:
                loopback = u.hostname == "localhost"
            if _flag(env, "SCOUT_LOCAL_ONLY") and not loopback:
                raise ValueError("SCOUT_LOCAL_ONLY requires a loopback endpoint")
            if u.scheme == "http" and not loopback and not _flag(env, "SCOUT_ALLOW_INSECURE_HTTP"):
                raise ValueError("Remote HTTP requires SCOUT_ALLOW_INSECURE_HTTP=1")
            # Treat localhost as a literal loopback address; do not depend on DNS.
            if u.hostname == "localhost" and _flag(env, "SCOUT_LOCAL_ONLY"):
                base = base.replace("://localhost", "://127.0.0.1", 1)
            # Do not reuse a vendor credential on a different host. The legacy
            # DeepSeek override remains an explicit historical exception.
            default_host = urlsplit(p["base_url"]).hostname
            if default_host and u.hostname != default_host and name != "deepseek":
                key = env.get("SCOUT_API_KEY", "").strip()
        if not local and _flag(env, "SCOUT_ALLOW_NO_KEY"):
            local = True  # permits an explicitly unauthenticated custom endpoint
        force = env.get("SCOUT_TOOL_CHOICE", p["force"]).strip().lower()
        reasoning = env.get("SCOUT_REASONING_MODE", p["reasoning"]).strip().lower()
        if force not in ("named", "prompt") or reasoning not in ("off", "deepseek", "openai"):
            raise ValueError("Invalid SCOUT_TOOL_CHOICE or SCOUT_REASONING_MODE")
        if p["protocol"] == "anthropic" and reasoning != "off":
            raise ValueError("Native Anthropic adapter does not accept OpenAI-style reasoning controls")
        timeout = float(env.get("SCOUT_TIMEOUT_SECONDS", "120"))
        max_output = int(env.get("SCOUT_MAX_OUTPUT_TOKENS", "0"))
        if not 0 < timeout <= 3600 or not 0 <= max_output <= 1_000_000:
            raise ValueError("Invalid Scout timeout or output-token limit")
        return cls(name, p["protocol"], base, model, check, key, key_env, local,
                   force, reasoning, timeout, max_output)

    def validate(self):
        if not self.base_url or not self.model or not self.check_model:
            raise RuntimeError("Configure SCOUT_BASE_URL (if needed) and SCOUT_MODEL / SCOUT_CHECK_MODEL")
        if not self.key and not self.local:
            raise RuntimeError(f"Set SCOUT_API_KEY or {self.key_env or 'SCOUT_ALLOW_NO_KEY=1'}")


def _obj(**kwargs):
    return SimpleNamespace(**kwargs)


def parse_tool_args(tool_calls, name, arguments):
    """Reject malformed calls locally, including providers without strict schemas."""
    schema = next((t["function"]["parameters"] for t in tool_calls
                   if t["function"]["name"] == name), None)
    if schema is None:
        raise ValueError("Unknown or unavailable tool")
    try:
        payload = json.loads(arguments)
    except (TypeError, ValueError):
        raise ValueError("Tool arguments must be valid JSON") from None
    if not isinstance(payload, dict) or not Draft7Validator(schema).is_valid(payload):
        raise ValueError("Tool arguments do not match the declared schema")
    return payload


class ProviderClient:
    def __init__(self, settings, transport=None):
        settings.validate()
        self.settings = settings
        self.http = httpx.Client(timeout=settings.timeout, follow_redirects=False,
                                 trust_env=False, transport=transport)

    def close(self):
        self.http.close()

    def complete(self, thinking="auto", *, force_tool=None, deterministic=False,
                 model, messages, tools=None, tool_choice=None):
        s = self.settings
        messages = copy.deepcopy(messages)
        tools = copy.deepcopy(tools or [])
        if force_tool:
            tools = [t for t in tools if t["function"]["name"] == force_tool]
            if not tools:
                raise ValueError("Requested forced tool is unavailable")
            if s.force == "prompt":
                messages.append({"role": "user", "content": f"Call {force_tool} now with the final structured result."})
        if s.protocol == "anthropic":
            path = "/messages"
            headers = {"anthropic-version": "2023-06-01", "x-api-key": s.key}
            payload = self._anthropic_request(model, messages, tools, force_tool, deterministic)
        else:
            path = "/chat/completions"
            headers = {}
            if s.key:
                if s.protocol == "azure":
                    headers["api-key"] = s.key
                else:
                    headers["Authorization"] = "Bearer " + s.key
            payload = self._chat_request(model, messages, tools, force_tool, thinking, deterministic, tool_choice)
        try:
            response = self.http.post(s.base_url + path, headers=headers, json=payload)
        except httpx.HTTPError:
            raise RuntimeError(f"{s.provider} request failed (network/timeout)") from None
        if not 200 <= response.status_code < 300:
            raise RuntimeError(f"{s.provider} request failed (HTTP {response.status_code})")
        try:
            data = response.json()
            return self._anthropic_response(data) if s.protocol == "anthropic" else self._chat_response(data)
        except (KeyError, IndexError, TypeError, ValueError, AttributeError):
            raise RuntimeError(f"{s.provider} returned an invalid or unsupported completion") from None

    def _chat_request(self, model, messages, tools, force_tool, thinking, deterministic, tool_choice):
        s = self.settings
        payload = {"model": model, "messages": messages, "stream": False}
        if tools:
            if s.provider not in ("deepseek", "openai", "azure"):
                for t in tools:
                    t["function"].pop("strict", None)
            payload["tools"] = tools
            if force_tool and s.force == "named":
                payload["tool_choice"] = {"type": "function", "function": {"name": force_tool}}
            elif s.force == "named" and tool_choice:
                payload["tool_choice"] = tool_choice
        if s.reasoning == "deepseek":
            if force_tool:
                payload["reasoning_effort"] = "none"
            elif thinking != "auto":
                payload["reasoning_effort"] = thinking
            if deterministic:
                payload["temperature"] = 0.0
        elif s.reasoning == "openai" and thinking != "auto":
            payload["reasoning_effort"] = "high" if thinking == "max" else thinking
        if s.max_output:
            payload["max_completion_tokens" if s.provider in ("openai", "azure") else "max_tokens"] = s.max_output
        return payload

    @staticmethod
    def _chat_response(data):
        raw = data["choices"][0]["message"]
        content = raw.get("content") or ""
        if not isinstance(content, str):
            raise ValueError("Expected text content")
        wire = {"role": "assistant", "content": content}
        for key in ("reasoning_content", "reasoning_details", "extra_content"):
            if raw.get(key) is not None:
                wire[key] = raw[key]
        calls = []
        for tc in raw.get("tool_calls") or []:
            f = tc["function"]
            if not tc.get("id") or not f.get("name"):
                raise ValueError("Missing tool call identifier")
            arguments = f.get("arguments")
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments)
            if not isinstance(arguments, str):
                raise ValueError("Invalid tool arguments")
            call = {"id": tc["id"], "type": "function", "function": {"name": f["name"], "arguments": arguments}}
            if tc.get("extra_content") is not None:
                call["extra_content"] = tc["extra_content"]
            wire.setdefault("tool_calls", []).append(call)
            calls.append(_obj(id=tc["id"], function=_obj(name=f["name"], arguments=arguments)))
        usage = data.get("usage") or {}
        hit = usage.get("prompt_cache_hit_tokens", (usage.get("prompt_tokens_details") or {}).get("cached_tokens"))
        miss = usage.get("prompt_cache_miss_tokens")
        if miss is None and hit is not None and isinstance(usage.get("prompt_tokens"), int):
            miss = max(0, usage["prompt_tokens"] - hit)
        return _obj(choices=[_obj(message=_obj(content=content, tool_calls=calls, wire=wire))],
                    usage=_obj(prompt_cache_hit_tokens=hit, prompt_cache_miss_tokens=miss))

    def _anthropic_request(self, model, messages, tools, force_tool, deterministic):
        system = []
        converted = []
        def add(role, blocks):
            if converted and converted[-1]["role"] == role:
                converted[-1]["content"].extend(blocks)
            else:
                converted.append({"role": role, "content": blocks})
        for m in messages:
            role = m["role"]
            if role == "system":
                system.append(m["content"])
            elif role == "tool":
                add("user", [{"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}])
            elif role == "assistant" and "_anthropic_blocks" in m:
                add("assistant", m["_anthropic_blocks"])
            else:
                add(role, [{"type": "text", "text": m["content"] or " "}])
        p = {"model": model, "messages": converted, "max_tokens": self.settings.max_output or 8192}
        if system:
            p["system"] = "\n\n".join(system)
        if tools:
            p["tools"] = [{"name": t["function"]["name"], "description": t["function"]["description"],
                           "input_schema": t["function"]["parameters"]} for t in tools]
            if force_tool and self.settings.force == "named":
                p["tool_choice"] = {"type": "tool", "name": force_tool}
            else:
                p["tool_choice"] = {"type": "auto"}
        if deterministic:
            p["temperature"] = 0
        return p

    @staticmethod
    def _anthropic_response(data):
        blocks = data["content"]
        text = "\n".join(b["text"] for b in blocks if b["type"] == "text")
        calls = [_obj(id=b["id"], function=_obj(name=b["name"], arguments=json.dumps(b["input"])))
                 for b in blocks if b["type"] == "tool_use"]
        wire = {"role": "assistant", "content": text, "_anthropic_blocks": blocks}
        usage = data.get("usage") or {}
        return _obj(choices=[_obj(message=_obj(content=text, tool_calls=calls, wire=wire))],
                    usage=_obj(prompt_cache_hit_tokens=usage.get("cache_read_input_tokens"),
                               prompt_cache_miss_tokens=usage.get("input_tokens")))
