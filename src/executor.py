from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    Task,
    TaskState,
    UnsupportedOperationError,
    InvalidRequestError,
)
from a2a.utils.errors import ServerError
from a2a.utils import (
    new_agent_text_message,
    new_task,
)

from agent import Agent

import logging

logger = logging.getLogger(__name__)


TERMINAL_STATES = {
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected,
}


class Executor(AgentExecutor):
    def __init__(
        self,
        data_dir: str = "data",
        use_gold_patches: bool = False,
        model_name: str = "gpt-4o",
        llm_api_base: str | None = None,
    ):
        self.agents: dict[str, Agent] = {}
        self.data_dir = data_dir
        self.use_gold_patches = use_gold_patches
        self.model_name = model_name
        self.llm_api_base = llm_api_base

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        msg = context.message
        if not msg:
            raise ServerError(error=InvalidRequestError(message="Missing message in request"))

        task = context.current_task
        if task and task.status.state in TERMINAL_STATES:
            raise ServerError(
                error=InvalidRequestError(
                    message=f"Task {task.id} already processed (state: {task.status.state})"
                )
            )

        if not task:
            task = new_task(msg)
            await event_queue.enqueue_event(task)

        context_id = task.context_id
        agent = self.agents.get(context_id)
        if not agent:
            agent = Agent(
                data_dir=self.data_dir,
                use_gold_patches=self.use_gold_patches,
                model_name=self.model_name,
                llm_api_base=self.llm_api_base,
            )
            self.agents[context_id] = agent

        updater = TaskUpdater(event_queue, task.id, context_id)

        await updater.start_work()
        try:
            await agent.run(msg, updater)
            if not updater._terminal_state_reached:
                logger.info(
                    "Executor completing task %s for context %s (terminal_reached=%s)",
                    task.id, context_id, updater._terminal_state_reached,
                )
                await updater.complete()
                logger.info("Executor completed task %s for context %s", task.id, context_id)
        except Exception as e:
            logger.error("Task failed with agent error: %s", e)
            await updater.failed(
                new_agent_text_message(f"Agent error: {e}", context_id=context_id, task_id=task.id)
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())
