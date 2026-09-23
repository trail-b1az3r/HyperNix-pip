"""HyperLink's default system prompt when hyperchat answers (0.72.6.rc2).

Checked where it matters: in the messages the model is actually sent.
It goes first, only on this server's own runner, under everything the
person wrote, and it can be turned off.
"""
from __future__ import annotations

import datetime

import pytest

from hypernix.hyperlink.default_prompt import DEFAULT_PROMPT, default_prompt, default_prompt_for
from hypernix.hyperlink.preferences import Preferences, system_prompt_for


class TestThePrompt:
    def test_it_is_more_than_three_paragraphs(self):
        paragraphs = [p for p in DEFAULT_PROMPT.split("\n\n") if p.strip()]
        assert len(paragraphs) > 3

    def test_it_carries_the_date(self):
        text = default_prompt(today=datetime.date(2026, 9, 3))
        assert "Thursday 3 September 2026" in text
        assert "{today}" not in text

    @pytest.mark.parametrize("point", [
        "running on the person's own computer",   # where it is
        "read on a phone",                         # who reads it
        "fenced blocks",                           # how code shows
        "Never make up",                           # no invention
        "only when it is actually offered",        # tools
        "theirs win",                              # the person's prompt wins
    ])
    def test_it_says_what_the_setting_needs(self, point):
        assert point in DEFAULT_PROMPT


class Config:
    hyperlink_default_prompt = True


class Runner:
    is_hypernix = True


class LMStudio:
    is_hypernix = False


class TestWhenItApplies:
    def test_on_the_runner(self):
        assert default_prompt_for(Config(), Runner()).startswith("You are a language model")

    def test_not_on_lm_studio(self):
        assert default_prompt_for(Config(), LMStudio()) == ""

    def test_not_when_turned_off(self):
        off = Config()
        off.hyperlink_default_prompt = False
        assert default_prompt_for(off, Runner()) == ""

    def test_it_goes_first_and_the_person_comes_after(self):
        prefs = Preferences(owner="me", display_name="Sam", system_prompt="Answer in French.")
        composed = system_prompt_for(prefs, session_prompt="This chat is about bread.",
                                     memory_block="Sam is vegetarian.", default="DEFAULT")
        order = [composed.index(x) for x in
                 ("DEFAULT", "Sam", "Answer in French.", "about bread", "vegetarian")]
        assert order == sorted(order)


# ---------------------------------------------------------------------------
# Through the API: what the model is sent
# ---------------------------------------------------------------------------

pytest.importorskip("fastapi")


class Loaded:
    model_id = "local-gguf"


class FakeRunner:
    base_url = "http://127.0.0.1:1"
    current = Loaded()


class Recorder:
    base_url = "http://127.0.0.1:1/v1"

    def __init__(self):
        self.sent = []

    def chat(self, messages, *, model=None, tools=None, **kwargs):
        system = str(messages[0].get("content", "")) if messages else ""
        if "You name conversations" not in system:
            self.sent.append(messages)
        return {"choices": [{"message": {"role": "assistant", "content": "ok"},
                             "finish_reason": "stop"}], "model": "local-gguf", "usage": {}}


@pytest.fixture
def api(tmp_path, monkeypatch):
    from conftest import clear_t1_config
    from fastapi.testclient import TestClient

    from hypernix.hyperlink import inference

    clear_t1_config(monkeypatch)
    monkeypatch.setenv("T1_DB_PATH", str(tmp_path / "t.sqlite3"))
    monkeypatch.setenv("T1_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("T1_TRUSTED_NETWORK", "1")
    recorder = Recorder()
    monkeypatch.setattr(inference, "_openai_client", lambda base_url, timeout=300.0: recorder)

    def make(*, runner=True, **env):
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        from hypernix.t1api.app import create_app

        app = create_app()
        if runner:
            app.state.t1_runner = FakeRunner()
        return TestClient(app, client=("192.168.1.9", 5000)), recorder
    return make


def _ask(client, text="hello"):
    session = client.post("/hyperlink/sessions", json={"title": "t"}).json()["session"]["session_id"]
    got = client.post(f"/hyperlink/sessions/{session}/chat", json={"content": text})
    assert got.status_code == 200, got.text
    return session


class TestThroughTheAPI:
    def test_the_runner_gets_it_first(self, api):
        client, model = api(T1_LMSTUDIO_ENABLED="0")
        _ask(client)
        system = model.sent[-1][0]
        assert system["role"] == "system"
        assert system["content"].startswith("You are a language model running on the person's own computer")

    def test_the_persons_prompt_follows_it(self, api):
        client, model = api(T1_LMSTUDIO_ENABLED="0")
        client.patch("/hyperlink/preferences", json={"system_prompt": "Always answer in French."})
        _ask(client)
        system = model.sent[-1][0]["content"]
        assert system.index("You are a language model") < system.index("Always answer in French.")
        # One system message, not two.
        assert sum(1 for m in model.sent[-1] if m["role"] == "system") == 1

    def test_lm_studio_does_not_get_it(self, api):
        client, model = api(runner=False, T1_LMSTUDIO_ENABLED="1")
        _ask(client)
        assert all("You are a language model running" not in str(m.get("content"))
                   for m in model.sent[-1])

    def test_it_can_be_turned_off(self, api):
        client, model = api(T1_LMSTUDIO_ENABLED="0", T1_HYPERLINK_DEFAULT_PROMPT="0")
        _ask(client)
        assert all("You are a language model running" not in str(m.get("content"))
                   for m in model.sent[-1])

    def test_it_is_not_stored_in_the_chat(self, api):
        """It is composed per turn, so changing it never rewrites history."""
        client, _model = api(T1_LMSTUDIO_ENABLED="0")
        session = _ask(client)
        messages = client.get(f"/hyperlink/sessions/{session}/messages").json()["messages"]
        assert all("You are a language model running" not in m["content"] for m in messages)
