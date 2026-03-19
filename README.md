# SWE-bench Pro Purple Agent (mini-swe-agent)

An A2A coding agent that uses [mini-swe-agent](https://github.com/scaleapi/mini-swe-agent) to solve real-world software engineering problems from SWE-bench Pro.

Receives a problem statement and Docker image from the green agent, runs mini-swe-agent inside a sibling Docker container, and returns the generated patch.

## Quick start

```bash
# Build
docker build -t swe-bench-purple-agent .

# Run (gold patch mode — no API key needed)
docker run -d -p 9009:9009 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e USE_GOLD_PATCHES=true \
  swe-bench-purple-agent --host 0.0.0.0 --port 9009

# Run (real mode)
docker run -d -p 9009:9009 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e MODEL_NAME=openrouter/deepseek/deepseek-v3.2 \
  -e OPENROUTER_API_KEY=your-key \
  swe-bench-purple-agent --host 0.0.0.0 --port 9009
```

## Tests

```bash
uv sync --extra test
uv run pytest -v --agent-url http://localhost:9009
```
