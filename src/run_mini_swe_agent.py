#!/usr/bin/env python3
"""
Subprocess runner for mini-swe-agent.

Invoked by agent.py as a child process so that litellm / httpx run in
complete isolation from the A2A event loop.

Usage:
    python run_mini_swe_agent.py --instance-file /tmp/instance.json

The instance JSON must contain:
    instance_id, problem_statement, docker_image, base_commit,
    model_name, config_path

Writes the patch (or empty string) to stdout as a JSON object:
    {"patch": "...", "exit_status": "..."}
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("mini-swe-agent-runner")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-file", required=True)
    args = parser.parse_args()

    with open(args.instance_file) as f:
        inst = json.load(f)

    instance_id = inst["instance_id"]
    problem_statement = inst["problem_statement"]
    docker_image = inst["docker_image"]
    model_name = inst["model_name"]
    config_path = inst["config_path"]

    import yaml
    from minisweagent.agents.default import AgentConfig, DefaultAgent
    from minisweagent.environments.docker import DockerEnvironment
    from minisweagent.models.litellm_model import LitellmModel

    with open(config_path) as f:
        swebench_config = yaml.safe_load(f)
    agent_cfg = swebench_config.get("agent", {})

    cmd_timeout = int(os.environ.get("MSWEA_CMD_TIMEOUT", 300))
    step_limit = int(os.environ.get("MSWEA_STEP_LIMIT", 500))
    cost_limit = float(os.environ.get("MSWEA_COST_LIMIT", 15.0))
    temperature = float(os.environ.get("MSWEA_TEMPERATURE", 0.0))
    llm_timeout = int(os.environ.get("MSWEA_LLM_TIMEOUT", 120))

    logger.info("Starting mini-swe-agent for %s (model=%s)", instance_id, model_name)

    env = DockerEnvironment(
        image=docker_image,
        cwd="/app",
        timeout=cmd_timeout,
        run_args=["--rm", "--entrypoint", ""],
        env={
            "PAGER": "cat",
            "MANPAGER": "cat",
            "LESS": "-R",
            "PIP_PROGRESS_BAR": "off",
            "TQDM_DISABLE": "1",
        },
    )

    try:
        model_kwargs = {
            "temperature": temperature,
            "drop_params": True,
            "timeout": llm_timeout,
        }
        extra_kwargs = {}
        if "anthropic" in model_name or "claude" in model_name:
            extra_kwargs["set_cache_control"] = "default_end"

        model = LitellmModel(
            model_name=model_name,
            model_kwargs=model_kwargs,
            **extra_kwargs,
        )

        agent = DefaultAgent(
            model=model,
            env=env,
            system_template=agent_cfg.get("system_template", AgentConfig.system_template),
            instance_template=agent_cfg.get("instance_template", AgentConfig.instance_template),
            action_observation_template=agent_cfg.get("action_observation_template", AgentConfig.action_observation_template),
            format_error_template=agent_cfg.get("format_error_template", AgentConfig.format_error_template),
            step_limit=agent_cfg.get("step_limit", step_limit),
            cost_limit=agent_cfg.get("cost_limit", cost_limit),
        )

        # Wrap step() to log each LLM exchange
        _orig_step = agent.step
        step_num = 0

        def _logging_step():
            nonlocal step_num
            step_num += 1
            logger.info("step %d | calls=%d cost=$%.4f", step_num, model.n_calls, model.cost)
            result = _orig_step()
            action = result.get("output", "")[:200]
            logger.info("step %d | action output: %s%s", step_num, action, "..." if len(result.get("output", "")) > 200 else "")
            return result

        agent.step = _logging_step

        exit_status, result_message, patch = agent.run(problem_statement)

        logger.info("mini-swe-agent finished: %s (patch length: %d)", exit_status, len(patch) if patch else 0)
        json.dump({"exit_status": str(exit_status), "patch": patch or ""}, sys.stdout)
    finally:
        env.cleanup()


if __name__ == "__main__":
    main()
