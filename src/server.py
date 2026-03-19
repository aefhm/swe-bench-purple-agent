import argparse
import os
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from executor import Executor


def main():
    parser = argparse.ArgumentParser(description="Run the purple (participant) A2A agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to data directory")
    parser.add_argument(
        "--use-gold-patches", action="store_true",
        default=os.environ.get("USE_GOLD_PATCHES", "").lower() in ("1", "true", "yes"),
        help="Use gold patches instead of running mini-swe-agent (for testing)",
    )
    parser.add_argument(
        "--model", type=str, default=os.environ.get("MODEL_NAME", "gpt-4o"),
        help="LLM model name for mini-swe-agent",
    )
    args = parser.parse_args()

    skill = AgentSkill(
        id="swe-bench-solver",
        name="SWE-bench Solver",
        description="Receives a SWE-bench problem and returns a patch that fixes it.",
        tags=["coding", "swe-bench", "patch"],
        examples=["Fix this issue in qutebrowser"],
    )

    mode = "gold patches" if args.use_gold_patches else f"mini-swe-agent ({args.model})"
    agent_card = AgentCard(
        name="SWE-bench Purple Agent",
        description=f"A2A coding agent that solves SWE-bench Pro instances using {mode}.",
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version="0.1.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(
            data_dir=args.data_dir,
            use_gold_patches=args.use_gold_patches,
            model_name=args.model,
        ),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
