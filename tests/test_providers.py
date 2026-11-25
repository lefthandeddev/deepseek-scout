import json
import tempfile
import unittest
from pathlib import Path

import httpx
from scout.providers import ProviderClient, Settings, REGISTRY, parse_tool_args
from scout.localtools import LocalTools
from scout.citations import verify_locations

TOOL = {"type": "function", "function": {"name": "read", "description": "Read",
    "strict": True, "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
    "required": ["path"], "additionalProperties": False}}}


class Providers(unittest.TestCase):
    def test_registry(self):
        for provider in REGISTRY:
            s = Settings.from_env({"SCOUT_PROVIDER": provider, "SCOUT_MODEL": "m"})
            self.assertEqual(s.provider, provider)
            self.assertEqual(s.model, "m")

    def test_legacy(self):
        s = Settings.from_env({"DEEPSEEK_API_KEY": "legacy", "DEEPSEEK_MODEL": "fast",
                               "DEEPSEEK_CHECK_MODEL": "checker", "DEEPSEEK_BASE_URL": "https://example.test"})
        self.assertEqual((s.provider, s.key, s.model, s.check_model), ("deepseek", "legacy", "fast", "checker"))
        self.assertNotIn("legacy", repr(s))

    def test_local_no_cloud_fallback(self):
        s = Settings.from_env({"SCOUT_PROVIDER": "ollama", "SCOUT_MODEL": "local",
                               "OPENAI_API_KEY": "must-not-leak", "SCOUT_LOCAL_ONLY": "1"})
        def handler(req):
            self.assertEqual(req.url.host, "127.0.0.1")
            self.assertNotIn("authorization", req.headers)
            body = json.loads(req.content)
            self.assertNotIn("reasoning_effort", body)
            self.assertNotIn("tool_choice", body)
            self.assertNotIn("strict", body["tools"][0]["function"])
            return httpx.Response(200, json={"choices": [{"message": {"content": "", "tool_calls": [
                {"id": "1", "function": {"name": "read", "arguments": '{"path":"a"}'}}
            ]}}]})
        c = ProviderClient(s, httpx.MockTransport(handler))
        try:
            m = c.complete("high", model=s.model, messages=[{"role": "user", "content": "x"}],
                           tools=[TOOL], force_tool="read").choices[0].message
            self.assertEqual(m.tool_calls[0].id, "1")
        finally:
            c.close()

    def test_deepseek_reasoning_roundtrip(self):
        s = Settings.from_env({"DEEPSEEK_API_KEY": "secret"})
        def handler(req):
            b = json.loads(req.content)
            self.assertEqual(b["reasoning_effort"], "none")
            self.assertEqual(b["tool_choice"]["function"]["name"], "read")
            self.assertTrue(b["tools"][0]["function"]["strict"])
            return httpx.Response(200, json={"choices": [{"message": {"content": "", "reasoning_content": "r",
                "tool_calls": [{"id": "a", "function": {"name": "read", "arguments": '{"path":"x"}'}}]}}],
                "usage": {"prompt_cache_hit_tokens": 4, "prompt_cache_miss_tokens": 2}})
        c = ProviderClient(s, httpx.MockTransport(handler))
        try:
            out = c.complete("high", model=s.model, messages=[{"role": "user", "content": "x"}],
                             tools=[TOOL], force_tool="read")
            self.assertEqual(out.choices[0].message.wire["reasoning_content"], "r")
            self.assertEqual(out.usage.prompt_cache_hit_tokens, 4)
        finally:
            c.close()

    def test_anthropic_roundtrip(self):
        s = Settings.from_env({"SCOUT_PROVIDER": "anthropic", "SCOUT_MODEL": "m", "ANTHROPIC_API_KEY": "k"})
        counter = 0
        blocks = [{"type": "text", "text": "checking"}, {"type": "tool_use", "id": "one", "name": "read", "input": {"path": "a"}},
                  {"type": "tool_use", "id": "two", "name": "read", "input": {"path": "b"}}]
        def handler(req):
            nonlocal counter
            counter += 1
            self.assertEqual(req.url.path, "/v1/messages")
            self.assertEqual(req.headers["x-api-key"], "k")
            b = json.loads(req.content)
            self.assertEqual(b["system"], "system")
            if counter == 1:
                return httpx.Response(200, json={"content": blocks})
            self.assertEqual(b["messages"][1]["content"], blocks)
            self.assertEqual([t["tool_use_id"] for t in b["messages"][2]["content"]], ["one", "two"])
            return httpx.Response(200, json={"content": [{"type": "text", "text": "done"}]})
        c = ProviderClient(s, httpx.MockTransport(handler))
        msgs = [{"role": "system", "content": "system"}, {"role": "user", "content": "go"}]
        try:
            m = c.complete(model="m", messages=msgs, tools=[TOOL]).choices[0].message
            msgs += [m.wire, {"role": "tool", "tool_call_id": "one", "content": "a"},
                     {"role": "tool", "tool_call_id": "two", "content": "b"}]
            self.assertEqual(c.complete(model="m", messages=msgs, tools=[TOOL]).choices[0].message.content, "done")
        finally:
            c.close()

    def test_validation_and_no_redirect(self):
        with self.assertRaises(ValueError):
            Settings.from_env({"SCOUT_PROVIDER": "ollama", "SCOUT_BASE_URL": "https://example.test", "SCOUT_LOCAL_ONLY": "1"})
        with self.assertRaises(ValueError):
            parse_tool_args([TOOL], "read", '{"path":9}')
        self.assertEqual(parse_tool_args([TOOL], "read", '{"path":"a"}'), {"path": "a"})
        s = Settings.from_env({"SCOUT_PROVIDER": "ollama", "SCOUT_MODEL": "m"})
        calls = []
        def handler(req):
            calls.append(str(req.url))
            return httpx.Response(307, headers={"location": "https://example.test"})
        c = ProviderClient(s, httpx.MockTransport(handler))
        try:
            with self.assertRaisesRegex(RuntimeError, "HTTP 307"):
                c.complete(model="m", messages=[])
            self.assertEqual(len(calls), 1)
        finally:
            c.close()

    def test_custom_host_key_not_reused(self):
        s = Settings.from_env({"SCOUT_PROVIDER": "openai", "SCOUT_MODEL": "m",
                               "SCOUT_BASE_URL": "https://example.test", "OPENAI_API_KEY": "vendor"})
        self.assertEqual(s.key, "")
        with self.assertRaises(RuntimeError):
            s.validate()

    def test_secret_and_symlink(self):
        with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as outside:
            root = Path(folder)
            (root / ".env").write_text("SECRET=secret")
            (root / "sample.py").write_text("-----BEGIN PRIVATE KEY-----\nPRIVATE-CONTENT\n-----END PRIVATE KEY-----\nprint(1)\n")
            tools = LocalTools(root)
            self.assertNotIn("PRIVATE-CONTENT", tools.read_file("sample.py", 2, 2))
            self.assertNotIn("PRIVATE-CONTENT", tools.grep("PRIVATE-CONTENT"))
            self.assertTrue(verify_locations(root, [{"path": ".env", "line": 1}]))
            try:
                (Path(outside) / "external.py").write_text("external secret")
                (root / "escape.py").symlink_to(Path(outside) / "external.py")
            except OSError:
                return
            self.assertNotIn("escape.py", tools.glob("*.py"))
            with self.assertRaises(ValueError):
                tools.read_file("escape.py")


if __name__ == "__main__":
    unittest.main()
