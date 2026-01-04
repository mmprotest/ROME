# ALE-lite

ALE-lite is a lightweight, local-first implementation of an “ALE-style” agentic training ecosystem inspired by arXiv:2512.24873. It combines:

- **ROCK**: a sandbox manager for safe tool execution.
- **iFlow**: a CLI agent runtime using OpenAI-compatible chat completions.
- **ROLL**: trajectory processing, IPA chunk scoring, and DPO dataset preparation.
- **TerminalBenchPro (TBP)**: a terminal-based evaluation harness.

This project is designed for local experimentation with OpenAI-compatible APIs (LM Studio, vLLM, Ollama OpenAI mode) and optional HF fine-tuning.

## Quick start

```bash
pip install -e .
```

Run a single task:

```bash
iflow run --task tasks/tbp_examples/basics_ls.yaml --config examples/configs/local_lmstudio.yaml
```

Run the TBP harness:

```bash
tbp run --tasks tasks/tbp_examples --config examples/configs/local_lmstudio.yaml --out runs/
```

Collect trajectories and score IPA chunks:

```bash
roll collect --runs runs/ --out datasets/raw.jsonl
roll ipa-score --in datasets/raw.jsonl --out datasets/ipa_scored.jsonl
roll make-dpo --in datasets/ipa_scored.jsonl --out datasets/dpo.jsonl
```

## OpenAI-compatible API configuration

Configuration files follow this schema:

```yaml
llm:
  base_url: "http://127.0.0.1:1234/v1"
  api_key: "lm-studio"
  model: "local-model-name"
  temperature: 0.2
  max_tokens: 1024
  timeout_s: 120
agent:
  max_steps: 40
  max_turns: 80
  tool_timeout_s: 60
  time_limit_s: 600
  context_max_tokens: 8000
sandbox:
  backend: auto # auto | docker | local
  image: "python:3.11-slim"
  network: false
  allowlist_paths: []
```

The client uses the official `openai` Python SDK (v1) and supports `base_url`, `api_key`, and configurable `model` values.

## ROCK sandbox isolation

ROCK provides a local sandbox backend with per-run temporary workspaces, resource limits, and best-effort network blocking. **LocalSandbox cannot fully disable network access**; it only clears proxy variables and should be treated as best-effort containment. For real containment control and network isolation, use DockerSandbox when Docker is available. The default `auto` backend selects Docker when available, otherwise Local. Use `rock doctor` to check backend availability.

## TBP harness

TBP task YAML schema includes:

- `id`, `description`, `goal`
- `setup_steps` (shell commands)
- `success_criteria` (`command_exit_code`, `file_contains`, `regex_in_stdout`, `unit_tests_pass`)
- `constraints` (`network`, `time_limit_s`, `max_steps`)

Each run produces a trajectory JSONL file with a stable schema and deterministic logging.

## IPA chunk scoring (approximation)

ALE-lite uses chunk-level credit assignment:

- Chunk boundaries include each assistant message, each tool call + observation, and optional `<micro-step>` tags inside assistant messages.
- Final reward `R` is derived from TBP score.
- Chunk rewards use a backward-weighted scheme (later chunks receive higher weight) with penalties for tool errors.
- Advantages are computed against the mean episode reward.

This IPA implementation is an approximation intended for local research, not a reproduction of the original paper.

## DPO dataset creation & training

`roll make-dpo` builds preference pairs by selecting the highest and lowest rewarded chunks per trajectory. Use `roll train-dpo` with a config file:

```yaml
model_name_or_path: /path/to/local/model
dataset_path: datasets/dpo.jsonl
output_dir: outputs/dpo_adapter
```

Training uses TRL + PEFT if installed (`pip install -e .[train]`).

## Development

Run linting and tests offline:

```bash
ruff check .
mypy src
pytest
```

## Lock files

This project uses standard PEP 621 metadata. Use your preferred tool (`uv`, `pip-tools`, or `poetry export`) to lock dependencies as needed.
