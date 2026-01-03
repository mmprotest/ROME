from __future__ import annotations

import json
from pathlib import Path

import typer

from ale_lite.roll.datasets import collect_runs, write_raw_dataset
from ale_lite.roll.ipa import assign_rewards, chunk_trajectory, write_ipa_scored
from ale_lite.roll.preference import make_dpo_records, write_dpo
from ale_lite.roll.train_dpo import load_train_config, train_dpo

app = typer.Typer(help="ROLL post-training pipeline")


@app.command()
def collect(runs: Path = typer.Option(..., "--runs"), out: Path = typer.Option(..., "--out")) -> None:
    trajectories = collect_runs(runs)
    write_raw_dataset(trajectories, out)
    typer.echo(f"Wrote {len(trajectories)} trajectories")


@app.command("ipa-score")
def ipa_score(input_path: Path = typer.Option(..., "--in"), out: Path = typer.Option(..., "--out")) -> None:
    records = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            events = record["events"]
            chunks = chunk_trajectory(events)
            final_reward = 0.0
            for event in events:
                if event["type"] == "outcome":
                    final_reward = float(event["payload"]["score"])
            scored = assign_rewards(chunks, final_reward)
            record["chunks"] = [
                {
                    "chunk": {
                        "state_summary": item["chunk"].state_summary,
                        "assistant_text": item["chunk"].assistant_text,
                        "tool_calls": item["chunk"].tool_calls,
                        "observations": item["chunk"].observations,
                        "outcome_features": item["chunk"].outcome_features,
                    },
                    "reward": item["reward"],
                    "advantage": item["advantage"],
                }
                for item in scored
            ]
            records.append(record)
    write_ipa_scored(records, out)
    typer.echo(f"Wrote {len(records)} ipa-scored records")


@app.command("make-dpo")
def make_dpo(input_path: Path = typer.Option(..., "--in"), out: Path = typer.Option(..., "--out")) -> None:
    with input_path.open("r", encoding="utf-8") as handle:
        ipa_records = [json.loads(line) for line in handle if line.strip()]
    dpo_records = make_dpo_records(ipa_records)
    write_dpo(dpo_records, out)
    typer.echo(f"Wrote {len(dpo_records)} dpo records")


@app.command("train-dpo")
def train_dpo_command(config: Path = typer.Option(..., "--config")) -> None:
    train_config = load_train_config(config)
    train_dpo(train_config)
