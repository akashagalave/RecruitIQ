
import argparse
import re
import time
from pathlib import Path

import dagshub
import mlflow

from hiring_agent.config.parameter_config import PARAMS_PATH, ROOT_DIR, params_config
from hiring_agent.app.prompts import EXECUTOR_PROMPT_NAME, PLANNER_PROMPT_NAME, RESPONDER_PROMPT_NAME, load_prompt
from hiring_agent.evals.eval_agent import GOLDENS_PATH, run_eval
from hiring_agent.evals.metrics import JUDGE_MODEL, TRACES_DIR
from hiring_agent.evals.report import metric_averages

REPO_OWNER = "akashagalaveaaa1"
REPO_NAME = "RecruitIQ"
TRACKING_URI = "https://dagshub.com/akashagalaveaaa1/RecruitIQ.mlflow"
EXPERIMENT_NAME = "hiring-agent"
PHASE_TAG = "noise_threshold"  

# the folders whose code is saved with every run
CODE_DIRS = ["app", "config", "evals"]


def flatten_params(params_dict: dict) -> dict:
    flat = {}
    for key, value in params_dict.items():
        items = flatten_params(value).items() if isinstance(value, dict) else [(key, value)]
        for flat_key, flat_value in items:
            # the section names are dropped, so two sections must not share a key
            if flat_key in flat:
                raise ValueError(f"Duplicate key found: {flat_key}")
            flat[flat_key] = flat_value
    return flat


def metric_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def prompt_versions_and_texts() -> tuple[dict, dict]:
    versions, texts = {}, {}
    for name in (PLANNER_PROMPT_NAME, EXECUTOR_PROMPT_NAME, RESPONDER_PROMPT_NAME):
        prompt = load_prompt(name)
        role = name.split("_")[2]  # hiring_agent_planner_prompt -> planner
        versions[f"{role}_prompt_version"] = "fallback" if prompt.is_fallback else prompt.version
        texts[name] = prompt.prompt
    return versions, texts


def return_code_files() -> list[Path]:
    package_dir = ROOT_DIR / "src" / "hiring_agent"
    code_files = [path for folder in CODE_DIRS for path in sorted((package_dir / folder).glob("*.py"))]
    return code_files + [PARAMS_PATH]


def code_artifact_dir(code_file: Path) -> str:
    return (Path("code") / code_file.parent.relative_to(ROOT_DIR)).as_posix()


def get_run_traces(since: float) -> list[Path]:
    return sorted(path for path in TRACES_DIR.glob("*.json") if path.stat().st_mtime >= since)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=int, nargs="+", help="run only these golden numbers (1-based)")
    args = parser.parse_args()

    dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        mlflow.set_tag("phase", PHASE_TAG)

        # params: params.yaml + the prompt versions actually fetched + the judge model
        prompt_versions, prompt_texts = prompt_versions_and_texts()
        params_dict = flatten_params(params_config.model_dump())
        mlflow.log_params({**params_dict, **prompt_versions, "judge_model": JUDGE_MODEL})

        # run the eval (agent + metrics on every golden); it also writes the md + csv report
        start = time.time()
        rows, report_md = run_eval(args.only)

        # metrics: average score per eval metric, rounded to 2 decimals like the report
        metrics = {metric_key(name): round(avg, 2) for name, avg in metric_averages(rows).items()}
        mlflow.log_metrics(metrics)

        # artifacts
        mlflow.log_artifact(GOLDENS_PATH, "golden_dataset")
        mlflow.log_artifact(report_md, "reports")
        mlflow.log_artifact(Path(report_md).with_suffix(".csv"), "reports")
        for trace_file in get_run_traces(since=start):
            mlflow.log_artifact(trace_file, "traces")
        for name, text in prompt_texts.items():
            mlflow.log_text(text, artifact_file=f"prompts/{name}.txt")
        for code_file in return_code_files():
            # keep the project layout (code/src/hiring_agent/app/...) so same-named files don't overwrite each other
            mlflow.log_artifact(code_file, code_artifact_dir(code_file))

    print(f"\nMLflow run: {run.info.run_name} ({run.info.run_id})")
    print(f"{TRACKING_URI}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")
