"""Run the existing DS-STAR pipeline over KramaBench workload files."""

import argparse
import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path

import yaml

from dsstar import DSConfig, DS_STAR_Agent


DOMAINS = (
    "archeology",
    "astronomy",
    "biomedical",
    "environment",
    "legal",
    "wildfire",
)


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def domain_data_root(kramabench_dir: Path, domain: str) -> Path:
    data_root = kramabench_dir / "data"
    for child in data_root.iterdir():
        if child.is_dir() and child.name.lower() == domain.lower():
            input_dir = child / "input"
            return input_dir if input_dir.is_dir() else child
    return data_root


def list_domain_files(kramabench_dir: Path, domain: str) -> list[str]:
    search_root = domain_data_root(kramabench_dir, domain)
    return sorted(str(path.resolve()) for path in search_root.rglob("*") if path.is_file())


def safe_run_id(domain: str, task_id: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{domain}_{task_id}")
    return value.strip("_.-")


def read_completed_ids(manifest_path: Path) -> set[str]:
    if not manifest_path.exists():
        return set()

    completed = set()
    with manifest_path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("status") == "success":
                completed.add(record["task_id"])
    return completed


def append_manifest(manifest_path: Path, record: dict) -> None:
    with manifest_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_summary_csv(manifest_path: Path, summary_path: Path) -> None:
    records = []
    with manifest_path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    latest = {record["task_id"]: record for record in records}
    fields = [
        "domain",
        "task_id",
        "status",
        "run_id",
        "output_file",
        "final_result",
        "expected_answer",
        "answer_type",
        "runtime_seconds",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "error",
        "finished_at",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(latest.values())


def print_domain_summary(manifest_path: Path, domain: str, data_file_count: int) -> None:
    latest = {}
    with manifest_path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("domain") == domain:
                latest[record["task_id"]] = record

    records = list(latest.values())
    runtime = sum(record.get("runtime_seconds", 0) or 0 for record in records)
    input_tokens = sum(record.get("input_tokens", 0) or 0 for record in records)
    output_tokens = sum(record.get("output_tokens", 0) or 0 for record in records)
    succeeded = sum(record.get("status") == "success" for record in records)
    failed = sum(record.get("status") == "failed" for record in records)

    print("\n" + "=" * 60)
    print(f"DOMAIN SUMMARY: {domain}")
    print(f"Data files:     {data_file_count:,}")
    print(f"Tasks recorded: {len(records):,}")
    print(f"Succeeded:      {succeeded:,}")
    print(f"Failed:         {failed:,}")
    print(f"Runtime:        {runtime:,.2f} seconds")
    print(f"Input tokens:   {input_tokens:,}")
    print(f"Output tokens:  {output_tokens:,}")
    print(f"Total tokens:   {input_tokens + output_tokens:,}")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full DS-STAR pipeline for KramaBench tasks."
    )
    parser.add_argument("--kramabench-dir", required=True, type=Path)
    parser.add_argument("--domain", choices=(*DOMAINS, "all"), default="all")
    parser.add_argument("--task-id", help="Run only one exact task ID")
    parser.add_argument("--start", type=int, default=0, help="Task index to start from")
    parser.add_argument("--limit", type=int, help="Maximum tasks per selected domain")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("kramabench_output"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kramabench_dir = args.kramabench_dir.resolve()
    if not (kramabench_dir / "data").is_dir():
        raise FileNotFoundError(f"KramaBench data folder not found: {kramabench_dir / 'data'}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = (args.output_dir / "runs").resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"
    summary_path = args.output_dir / "summary.csv"
    completed_ids = read_completed_ids(manifest_path)
    defaults = load_config(args.config)
    domains = DOMAINS if args.domain == "all" else (args.domain,)

    for domain in domains:
        workload_path = kramabench_dir / "workload" / f"{domain}.json"
        if not workload_path.exists():
            raise FileNotFoundError(f"Workload not found: {workload_path}")
        tasks = json.loads(workload_path.read_text(encoding="utf-8"))
        if args.task_id:
            tasks = [task for task in tasks if task["id"] == args.task_id]
        tasks = tasks[args.start :]
        if args.limit is not None:
            tasks = tasks[: args.limit]

        data_files = list_domain_files(kramabench_dir, domain)
        if not data_files:
            raise FileNotFoundError(
                f"No input files found for domain {domain}: "
                f"{domain_data_root(kramabench_dir, domain)}"
            )
        analysis_cache_dir = (args.output_dir / "analysis_cache" / domain).resolve()

        for task in tasks:
            task_id = task["id"]
            if task_id in completed_ids:
                print(f"[SKIP] {task_id} already completed")
                continue

            run_id = safe_run_id(domain, task_id)
            print(f"[RUN] {task_id}: all {len(data_files)} {domain} data file(s)")
            started_at = time.perf_counter()
            config = DSConfig(
                run_id=run_id,
                max_refinement_rounds=defaults.get("max_refinement_rounds", 5),
                api_key=defaults.get("api_key"),
                model_name=defaults.get("model_name"),
                interactive=False,
                auto_debug=defaults.get("auto_debug", True),
                debug_attempts=defaults.get("debug_attempts", 3),
                execution_timeout=defaults.get("execution_timeout", 60),
                preserve_artifacts=defaults.get("preserve_artifacts", True),
                runs_dir=str(runs_dir),
                analysis_cache_dir=str(analysis_cache_dir),
                agent_models=defaults.get("agent_models", {}),
            )

            agent = None
            try:
                agent = DS_STAR_Agent(config)
                result = agent.run_pipeline(task["query"], data_files)
                record = {
                    "domain": domain,
                    "task_id": task_id,
                    "status": "success",
                    "run_id": result["run_id"],
                    "output_file": result["output_file"],
                    "final_result": result["final_result"],
                    "expected_answer": task.get("answer"),
                    "answer_type": task.get("answer_type", ""),
                    "error": "",
                    "finished_at": datetime.now().isoformat(),
                }
                completed_ids.add(task_id)
                print(f"[DONE] {task_id}")
            except Exception as exc:
                record = {
                    "domain": domain,
                    "task_id": task_id,
                    "status": "failed",
                    "run_id": run_id,
                    "output_file": "",
                    "final_result": "",
                    "expected_answer": task.get("answer"),
                    "answer_type": task.get("answer_type", ""),
                    "error": str(exc),
                    "finished_at": datetime.now().isoformat(),
                }
                print(f"[FAILED] {task_id}: {exc}")

            usage = agent.get_model_usage() if agent else {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }
            record.update(usage)
            record["runtime_seconds"] = round(time.perf_counter() - started_at, 3)
            print(
                f"[METRICS] {task_id}: "
                f"runtime={record['runtime_seconds']:.3f}s, "
                f"input={record['input_tokens']:,}, "
                f"output={record['output_tokens']:,}, "
                f"total={record['total_tokens']:,}"
            )
            append_manifest(manifest_path, record)
            write_summary_csv(manifest_path, summary_path)

        print_domain_summary(manifest_path, domain, len(data_files))


if __name__ == "__main__":
    main()
