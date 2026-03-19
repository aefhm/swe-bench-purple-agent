"""
Purple agent — SWE-bench Pro participant (coding agent).

Uses mini-swe-agent to actually solve issues inside Docker containers.
Falls back to gold patches if --use-gold-patches is set (for pipeline testing).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger

logger = logging.getLogger("purple-agent")


class Agent:
    def __init__(
        self,
        data_dir: str = "data",
        use_gold_patches: bool = False,
        model_name: str = "gpt-4o",
    ):
        self.messenger = Messenger()
        self.data_dir = data_dir
        self.use_gold_patches = use_gold_patches
        self.model_name = model_name
        self._gold_patches: dict[str, str] | None = None

    @property
    def gold_patches(self) -> dict[str, str]:
        """Lazy-load gold patches keyed by instance_id from instances.jsonl."""
        if self._gold_patches is None:
            path = Path(self.data_dir) / "instances.jsonl"
            if path.exists():
                self._gold_patches = {}
                with open(path) as f:
                    for line in f:
                        if not line.strip():
                            continue
                        d = json.loads(line)
                        if d.get("gold_patch"):
                            self._gold_patches[d["instance_id"]] = d["gold_patch"]
            else:
                self._gold_patches = {}
        return self._gold_patches

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)

        await updater.update_status(
            TaskState.working, new_agent_text_message("Parsing problem...")
        )

        # Parse the problem payload from the green agent
        try:
            problem = json.loads(input_text)
            instance_id = problem["instance_id"]
            problem_statement = problem["problem_statement"]
            docker_image = problem["docker_image"]
            base_commit = problem.get("base_commit", "")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=f"Error: could not parse problem payload: {e}"))],
                name="Error",
            )
            return

        # ---- Gold patch mode (for testing) ----
        if self.use_gold_patches:
            patch = self.gold_patches.get(instance_id)
            if patch is None:
                await updater.add_artifact(
                    parts=[Part(root=TextPart(text=f"Error: no gold patch for {instance_id}"))],
                    name="Error",
                )
                return

            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Returning gold patch for {instance_id}"),
            )
            result = json.dumps({"instance_id": instance_id, "patch": patch})
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=result))],
                name="Patch",
            )
            return

        # ---- Real mode: use mini-swe-agent ----
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"Solving {instance_id} with mini-swe-agent (image: {docker_image})..."
            ),
        )

        try:
            patch = await self._run_mini_swe_agent(
                instance_id=instance_id,
                problem_statement=problem_statement,
                docker_image=docker_image,
                base_commit=base_commit,
            )
        except Exception as e:
            logger.exception(f"mini-swe-agent failed for {instance_id}")
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=f"Error: mini-swe-agent failed: {e}"))],
                name="Error",
            )
            return

        if not patch:
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=f"Error: mini-swe-agent produced no patch for {instance_id}"))],
                name="Error",
            )
            return

        result = json.dumps({"instance_id": instance_id, "patch": patch})
        await updater.add_artifact(
            parts=[Part(root=TextPart(text=result))],
            name="Patch",
        )

    async def _run_mini_swe_agent(
        self,
        *,
        instance_id: str,
        problem_statement: str,
        docker_image: str,
        base_commit: str,
    ) -> str | None:
        """Run mini-swe-agent in a Docker container to solve the problem.

        This runs synchronously (mini-swe-agent is sync) so we run it in a
        thread to avoid blocking the event loop.
        """
        import asyncio

        return await asyncio.to_thread(
            self._run_mini_swe_agent_sync,
            instance_id=instance_id,
            problem_statement=problem_statement,
            docker_image=docker_image,
            base_commit=base_commit,
        )

    def _run_mini_swe_agent_sync(
        self,
        *,
        instance_id: str,
        problem_statement: str,
        docker_image: str,
        base_commit: str,
    ) -> str | None:
        """Synchronous core: create DockerEnvironment + DefaultAgent, run, return patch."""
        import yaml
        from minisweagent.agents.default import AgentConfig, DefaultAgent
        from minisweagent.config import get_config_path
        from minisweagent.environments.docker import DockerEnvironment
        from minisweagent.models.litellm_model import LitellmModel

        logger.info(f"Starting mini-swe-agent for {instance_id}")
        logger.info(f"  image: {docker_image}")
        logger.info(f"  model: {self.model_name}")

        # Load the SWE-bench config for proper prompts/templates
        # Prefer vendored config next to this file, fall back to installed package
        config_path = Path(__file__).resolve().parent.parent / "config" / "swebench.yaml"
        if not config_path.exists():
            config_path = get_config_path("swebench")
        with open(config_path) as f:
            swebench_config = yaml.safe_load(f)
        agent_cfg = swebench_config.get("agent", {})

        # SWE-bench Pro repos are cloned into /app (not /testbed).
        # The base images have ENTRYPOINT ["/bin/bash"], so we must
        # override it — otherwise `sleep 2h` gets passed as a bash
        # script arg and the container exits immediately.
        # Also override cwd to /app (swebench.yaml defaults to /testbed).
        cmd_timeout = int(os.environ.get("MSWEA_CMD_TIMEOUT", 300))
        step_limit = int(os.environ.get("MSWEA_STEP_LIMIT", 500))
        cost_limit = float(os.environ.get("MSWEA_COST_LIMIT", 15.0))
        temperature = float(os.environ.get("MSWEA_TEMPERATURE", 0.0))

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
            model_kwargs = {"temperature": temperature, "drop_params": True}
            extra_kwargs = {}
            if "anthropic" in self.model_name or "claude" in self.model_name:
                extra_kwargs["set_cache_control"] = "default_end"

            model = LitellmModel(
                model_name=self.model_name,
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

            exit_status, result_message, patch = agent.run(problem_statement)

            logger.info(f"mini-swe-agent finished for {instance_id}: {exit_status}")
            logger.info(f"  patch length: {len(patch) if patch else 0}")

            return patch if patch else None
        finally:
            env.cleanup()
