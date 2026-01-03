from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

from ale_lite.api.openai_client import OpenAIChatClient, OpenAIConfig
from ale_lite.iflow.agent import Agent, AgentConfig
from ale_lite.iflow.prompts import TaskSpec as AgentTaskSpec
from ale_lite.iflow.trajectory import TrajectoryWriter, outcome_event
from ale_lite.rock.local_sandbox import LocalSandbox
from ale_lite.rock.sandbox import SandboxConfig
from ale_lite.tbp.scoring import ScoreResult, evaluate
from ale_lite.tbp.tasks import TaskSpec


@dataclass
class RunResult:
    task_id: str
    success: bool
    score: float
    trajectory_path: Path


def load_config(path: Path) -> Dict[str, Dict[str, object]]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def run_task(
    task: TaskSpec,
    config: Dict[str, Dict[str, object]],
    out_dir: Path,
    agent_factory: Optional[Callable[[LocalSandbox, TrajectoryWriter], Agent]] = None,
) -> RunResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    sandbox_config = SandboxConfig(
        network_enabled=bool(task.constraints.get("network", False)),
    )
    sandbox = LocalSandbox(sandbox_config)
    sandbox.create_workspace()
    trajectory_path = out_dir / f"{task.id}_trajectory.jsonl"
    trajectory = TrajectoryWriter(trajectory_path)

    for step in task.setup_steps:
        sandbox.run_command(step, timeout_s=30)

    if agent_factory is None:
        llm_config = config["llm"]
        agent_cfg = config.get("agent", {})
        client = OpenAIChatClient(
            OpenAIConfig(
                base_url=str(llm_config["base_url"]),
                api_key=str(llm_config["api_key"]),
                model=str(llm_config["model"]),
                temperature=float(llm_config.get("temperature", 0.2)),
                max_tokens=int(llm_config.get("max_tokens", 1024)),
                timeout_s=float(llm_config.get("timeout_s", 120)),
            )
        )
        agent = Agent(
            client,
            sandbox,
            trajectory,
            AgentConfig(
                max_steps=int(agent_cfg.get("max_steps", 40)),
                max_turns=int(agent_cfg.get("max_turns", 80)),
                tool_timeout_s=float(agent_cfg.get("tool_timeout_s", 60)),
                context_max_tokens=int(agent_cfg.get("context_max_tokens", 8000)),
                memory_items=int(agent_cfg.get("memory_items", 20)),
            ),
        )
    else:
        agent = agent_factory(sandbox, trajectory)

    agent_task = AgentTaskSpec(goal=task.goal, evaluation=task.evaluation_text)
    result = agent.run(agent_task)

    score_result = evaluate(sandbox, task.success_criteria)
    trajectory.log("outcome", outcome_event(score_result.success, score_result.score, result.reason))
    sandbox.teardown()

    return RunResult(
        task_id=task.id,
        success=score_result.success,
        score=score_result.score,
        trajectory_path=trajectory_path,
    )
