"""
Purple agent — SWE-bench Pro participant (coding agent).

Uses mini-swe-agent to actually solve issues inside Docker containers.
Falls back to gold patches if --use-gold-patches is set (for pipeline testing).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import subprocess
import tempfile
import threading
from pathlib import Path

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger

logger = logging.getLogger("purple-agent")

# Resolve once at import time
_RUNNER_SCRIPT = str(Path(__file__).resolve().parent / "run_mini_swe_agent.py")
_DEFAULT_CONFIG = str(Path(__file__).resolve().parent.parent / "config" / "swebench.yaml")

_SENTINEL = None  # marks end of stderr stream


class Agent:
    def __init__(
        self,
        data_dir: str = "data",
        use_gold_patches: bool = False,
        model_name: str = "gpt-4o",
        llm_api_base: str | None = None,
    ):
        self.messenger = Messenger()
        self.data_dir = data_dir
        self.use_gold_patches = use_gold_patches
        self.model_name = model_name
        self.llm_api_base = llm_api_base
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
                updater=updater,
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
        updater: TaskUpdater,
    ) -> str | None:
        """Run mini-swe-agent as a subprocess, fully isolated from the A2A event loop.

        stderr streams live to both the server log and as A2A status updates
        back to the green agent.  stdout is captured for the JSON result.
        """
        from minisweagent.config import get_config_path

        config_path = _DEFAULT_CONFIG
        if not Path(config_path).exists():
            config_path = str(get_config_path("swebench"))

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix=f"mswea-{instance_id}-", delete=False
        ) as f:
            json.dump({
                "instance_id": instance_id,
                "problem_statement": problem_statement,
                "docker_image": docker_image,
                "base_commit": base_commit,
                "model_name": self.model_name,
                "llm_api_base": self.llm_api_base,
                "config_path": config_path,
            }, f)
            instance_file = f.name

        try:
            timeout = int(os.environ.get("MSWEA_SUBPROCESS_TIMEOUT", 3600))

            # Thread-safe queue: subprocess thread pushes lines, async task consumes them
            log_queue: queue.Queue[str | None] = queue.Queue()

            # Start subprocess in a background thread
            sub_future: asyncio.Future[tuple[str, int]] = asyncio.get_event_loop().run_in_executor(
                None,
                self._run_subprocess,
                instance_file,
                timeout,
                log_queue,
            )

            # Forward log lines as A2A status updates while subprocess runs
            while True:
                try:
                    line = await asyncio.to_thread(log_queue.get, timeout=0.5)
                except Exception:
                    # queue.get timed out — check if subprocess is done
                    if sub_future.done():
                        break
                    continue

                if line is _SENTINEL:
                    break

                logger.info("[runner] %s", line)
                # Send step lines back to green agent as status updates
                if "step " in line:
                    # Extract the meaningful part after the log prefix
                    # e.g. "2026-03-24 ... INFO step 3 | calls=2 cost=$0.05"
                    parts = line.split(" INFO ", 1)
                    status_text = parts[1] if len(parts) > 1 else line
                    await updater.update_status(
                        TaskState.working,
                        new_agent_text_message(status_text),
                    )

            stdout, returncode = await sub_future

            if returncode != 0:
                raise RuntimeError(
                    f"mini-swe-agent subprocess exited with code {returncode}"
                )

            output = json.loads(stdout)
            patch = output.get("patch", "")

            logger.info(f"mini-swe-agent finished for {instance_id}: {output.get('exit_status')}")
            logger.info(f"  patch length: {len(patch)}")

            return patch if patch else None
        finally:
            Path(instance_file).unlink(missing_ok=True)

    @staticmethod
    def _run_subprocess(
        instance_file: str,
        timeout: int,
        log_queue: queue.Queue[str | None],
    ) -> tuple[str, int]:
        """Run the runner script, pushing stderr lines into the queue."""
        proc = subprocess.Popen(
            ["python", _RUNNER_SCRIPT, "--instance-file", instance_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ},
        )

        try:
            assert proc.stderr is not None
            for line in proc.stderr:
                log_queue.put(line.rstrip("\n"))

            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise
        finally:
            log_queue.put(_SENTINEL)

        assert proc.stdout is not None
        stdout = proc.stdout.read()
        return stdout, proc.returncode
