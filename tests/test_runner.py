import json
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_mini_swe_agent as runner


def test_runner_uses_llm_api_base(monkeypatch, tmp_path, capsys):
    captured: dict[str, object] = {}

    class FakeAgentConfig:
        system_template = "system"
        instance_template = "instance"
        action_observation_template = "action"
        format_error_template = "error"

    class FakeDefaultAgent:
        def __init__(self, **kwargs):
            captured["agent_init"] = kwargs
            self.step = lambda: {"output": ""}

        def run(self, problem_statement):
            captured["problem_statement"] = problem_statement
            return "completed", "done", "diff --git a/foo b/foo"

    class FakeDockerEnvironment:
        def __init__(self, **kwargs):
            captured["docker_env_init"] = kwargs

        def cleanup(self):
            captured["cleanup_called"] = True

    class FakeLitellmModel:
        def __init__(self, **kwargs):
            captured["model_init"] = kwargs
            self.n_calls = 0
            self.cost = 0.0

    monkeypatch.setitem(sys.modules, "minisweagent", types.ModuleType("minisweagent"))
    monkeypatch.setitem(
        sys.modules, "minisweagent.agents", types.ModuleType("minisweagent.agents")
    )
    monkeypatch.setitem(
        sys.modules,
        "minisweagent.agents.default",
        types.SimpleNamespace(AgentConfig=FakeAgentConfig, DefaultAgent=FakeDefaultAgent),
    )
    monkeypatch.setitem(
        sys.modules,
        "minisweagent.environments",
        types.ModuleType("minisweagent.environments"),
    )
    monkeypatch.setitem(
        sys.modules,
        "minisweagent.environments.docker",
        types.SimpleNamespace(DockerEnvironment=FakeDockerEnvironment),
    )
    monkeypatch.setitem(
        sys.modules,
        "minisweagent.models",
        types.ModuleType("minisweagent.models"),
    )
    monkeypatch.setitem(
        sys.modules,
        "minisweagent.models.litellm_model",
        types.SimpleNamespace(LitellmModel=FakeLitellmModel),
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text("agent: {}\n")
    instance_path = tmp_path / "instance.json"
    instance_path.write_text(
        json.dumps(
            {
                "instance_id": "instance-1",
                "problem_statement": "Fix the bug",
                "docker_image": "example/image:latest",
                "model_name": "gpt-4o",
                "llm_api_base": "http://127.0.0.1:4010",
                "config_path": str(config_path),
            }
        )
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_mini_swe_agent.py", "--instance-file", str(instance_path)],
    )

    runner.main()

    model_init = captured["model_init"]
    assert model_init["model_kwargs"]["base_url"] == "http://127.0.0.1:4010"
    assert captured["cleanup_called"] is True

    result = json.loads(capsys.readouterr().out)
    assert result["patch"] == "diff --git a/foo b/foo"
