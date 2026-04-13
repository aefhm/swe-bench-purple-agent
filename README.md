# SWE-bench Pro Purple Agent (mini-swe-agent)

A reference coding agent for [SWE-bench Pro](https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro). **Fork this repo to compete.**

Uses [mini-swe-agent](https://github.com/scaleapi/mini-swe-agent) to solve real-world software engineering problems. Receives a problem statement and Docker image from the green agent, runs mini-swe-agent inside a sibling Docker container, and returns the generated patch.

## Quick start

```bash
# Build
docker build -t swe-bench-purple-agent .

# Run
docker run -d -p 9009:9009 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e MODEL_NAME=gpt-4o \
  -e OPENAI_API_KEY=your-key \
  swe-bench-purple-agent --host 0.0.0.0 --port 9009

# Run against an OpenAI-compatible proxy provided by Amber
docker run -d -p 9009:9009 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e MODEL_NAME=gpt-4o \
  -e LLM_API_BASE=http://host.docker.internal:4010 \
  swe-bench-purple-agent --host 0.0.0.0 --port 9009
```

## Configuration

The agent's behavior is controlled at three levels (highest priority first):

1. **`config/swebench.yaml`** — `step_limit`, `cost_limit`, system prompt templates
2. **Environment variables** — `MODEL_NAME`, API keys, `MSWEA_*` tunables
3. **CLI defaults** in `src/server.py`

Key settings in `config/swebench.yaml`:

| Setting | Default | Description |
|---|---|---|
| `step_limit` | 25 | Max LLM turns per instance |
| `cost_limit` | 2.0 | Max USD spend per instance |

Key environment variables:

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `gpt-4o` | LiteLLM model string |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `OPENROUTER_API_KEY` | — | OpenRouter API key |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `GEMINI_API_KEY` | — | Gemini API key |
| `MSWEA_MAX_TOKENS` | 4096 | Max tokens per LLM response |
| `MSWEA_CMD_TIMEOUT` | 300 | Shell command timeout (seconds) |
| `MSWEA_SUBPROCESS_TIMEOUT` | 900 | Total subprocess timeout (seconds) |

## Customizing for competition

1. Fork this repo
2. Modify `config/swebench.yaml` to set your preferred model and limits
3. Build and push your Docker image to a registry
4. Update the image reference in your leaderboard fork's `scenario.json5`
5. Submit via the leaderboard repo

## Tests

```bash
uv sync --extra test
uv run pytest -v --agent-url http://localhost:9009
```
