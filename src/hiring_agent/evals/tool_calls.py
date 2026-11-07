from deepeval.test_case import ToolCall
from langchain_core.callbacks import BaseCallbackHandler

FREE_TEXT_ARGS = {"notes", "subject", "body"}


def _is_lookup(name: str, args: dict) -> bool:
    return name == "list_files" or (name == "read_file" and args.get("path") == "candidates.csv")


def clean_tool_call(name: str, args: dict) -> ToolCall | None:
    args = args or {}
    if _is_lookup(name, args):
        return None

    kept = {k: v for k, v in args.items() if k not in FREE_TEXT_ARGS}
    if "nice_to_haves" in kept:
        skills = [s.strip() for s in str(kept["nice_to_haves"]).split(";") if s.strip()]
        kept["nice_to_haves"] = ";".join(sorted(skills))
    return ToolCall(name=name, input_parameters=kept)


def clean_tool_calls(calls: list[ToolCall]) -> list[ToolCall]:
    cleaned = [clean_tool_call(c.name, c.input_parameters) for c in calls or []]
    return [c for c in cleaned if c is not None]


class ToolCallRecorder(BaseCallbackHandler):
    def __init__(self):
        self.tool_calls: list[ToolCall] = []

    def on_tool_start(self, serialized, input_str, *, inputs=None, **kwargs):
        call = clean_tool_call(serialized["name"], inputs)
        if call is not None:
            self.tool_calls.append(call)
