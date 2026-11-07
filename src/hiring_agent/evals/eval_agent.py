
import argparse

from dotenv import load_dotenv

load_dotenv()

from deepeval.dataset import EvaluationDataset
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig
from deepeval.integrations.langchain import CallbackHandler
from deepeval.test_case import LLMTestCase

from hiring_agent.evals.adherence_judge import adherence_test_case
from hiring_agent.evals.metrics import (
    ADHERENCE_JUDGE,
    JUDGE_MODEL,
    PLAN_QUALITY_JUDGE,
    TOOL_CORRECTNESS,
    make_plan_adherence_judge,
    make_plan_quality_judge,
    make_tool_correctness,
    make_trace_metrics,
)
from hiring_agent.evals.plan_judge import plan_test_case
from hiring_agent.evals.report import print_summary, write_report
from hiring_agent.evals.tool_calls import ToolCallRecorder, clean_tool_calls
from hiring_agent.app import clients
from hiring_agent.app.agent import build_agent, run_agent
from hiring_agent.app.prompts import EXECUTOR_PROMPT_NAME, PLANNER_PROMPT_NAME, RESPONDER_PROMPT_NAME, load_prompt
from hiring_agent.app.workspace import ROOT_DIR, reset_workspace, workspace_snapshot

GOLDENS_PATH = ROOT_DIR / "goldens" / "goldens.json"


def load_goldens(only: list[int] | None):
    dataset = EvaluationDataset()
    dataset.add_goldens_from_json_file(str(GOLDENS_PATH))
    goldens = dataset.goldens
    if only:
        goldens = [goldens[n - 1] for n in only]  # 1-based, like the report
    return EvaluationDataset(goldens=goldens)


def run_metadata() -> dict:
    prompts = ", ".join(f"{n.split('_')[2]} v{load_prompt(n).version}" for n in (PLANNER_PROMPT_NAME, EXECUTOR_PROMPT_NAME, RESPONDER_PROMPT_NAME))
    return {
        "Planner model": clients.PLANNER_MODEL,
        "Executor / responder model": clients.EXECUTOR_MODEL,
        "Judge model": JUDGE_MODEL,
        "Langfuse prompts": prompts,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=int, nargs="+", help="run only these golden numbers (1-based)")
    args = parser.parse_args()

    dataset = load_goldens(args.only)
    agent = build_agent()
    meta = run_metadata()

    rows = []  # (golden, {metric name: metric}, final state); scores are filled in by DeepEval
    try:
        for golden in dataset.evals_iterator(
            async_config=AsyncConfig(run_async=False),  # one golden at a time: they share one workspace/
            display_config=DisplayConfig(inspect_after_run=False),
        ):
            reset_workspace()
            snapshot = workspace_snapshot()  # exactly what the planner is about to see

            # fresh metrics for every golden: a metric stores its own score and reason
            trace_metrics = make_trace_metrics()
            plan_judge = make_plan_quality_judge()
            adherence_judge = make_plan_adherence_judge()
            tool_correctness = make_tool_correctness()
            metrics = {**trace_metrics, PLAN_QUALITY_JUDGE: plan_judge, ADHERENCE_JUDGE: adherence_judge,
                       TOOL_CORRECTNESS: tool_correctness}

            recorder = ToolCallRecorder()  # collects this golden's tool calls for Tool Correctness
            state = run_agent(agent, golden.input, callbacks=[CallbackHandler(metrics=list(trace_metrics.values())), recorder], full=True)
            rows.append((golden, metrics, state))

            # these read the final state or the recorded tool calls, not the trace
            tools_test_case = LLMTestCase(
                input=golden.input,
                actual_output=state["response"],
                tools_called=recorder.tool_calls,
                expected_tools=clean_tool_calls(golden.expected_tools),  # same cleaning on both sides
            )
            for judge, test_case in (
                (plan_judge, plan_test_case(golden.input, state["plan"], snapshot)),
                (adherence_judge, adherence_test_case(golden.input, state["plan"], state["past_steps"])),
                (tool_correctness, tools_test_case),
            ):
                try:
                    judge.measure(test_case)
                except Exception as e:  # show it in the report instead of crashing the run
                    judge.error = f"{type(e).__name__}: {e}"
    finally:
        if rows:
            print_summary(rows)
            print("\nReport:", write_report(rows, "agent_eval", meta))
