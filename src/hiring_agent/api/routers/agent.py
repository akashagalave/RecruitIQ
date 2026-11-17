
from fastapi import APIRouter, HTTPException, status

from hiring_agent.app import workspace
from hiring_agent.app.agent import build_agent, run_agent
from hiring_agent.app.clients import logger

from hiring_agent.api.schemas import AgentRunResponse, Changes, TaskRequest

router = APIRouter(prefix="/agent", tags=["agent"])

agent = build_agent()

def take_snapshot() -> dict:
    """Every file path and every tracker id, taken before and after a run."""
    files = {path for paths in workspace.file_tree().values() for path in paths}
    tracker_ids = {row["candidate_id"] for row in workspace.read_tracker()}
    return {"files": files, "tracker_ids": tracker_ids}


def diff_snapshots(before: dict, after: dict) -> Changes:
    """Added, moved and deleted files plus new tracker rows."""
    added = sorted(after["files"] - before["files"])
    deleted = sorted(before["files"] - after["files"])

    # a file that left one folder and appeared in another with the same name was moved
    moved = []
    for old in list(deleted):
        name = old.split("/")[-1]
        new = next((path for path in added if path.split("/")[-1] == name), None)
        if new:
            moved.append((old, new))
            deleted.remove(old)
            added.remove(new)

    new_rows = len(after["tracker_ids"] - before["tracker_ids"])
    return Changes(added=added, moved=moved, deleted=deleted, new_rows=new_rows)


@router.post("/run", response_model=AgentRunResponse)
def run(request: TaskRequest) -> AgentRunResponse:

    before = take_snapshot()
    try:
        state = run_agent(agent, request.task, full=True)
    except Exception as error:
        logger.error(f"[agent] run failed: {type(error).__name__}: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to run the agent.",
        ) from error
    return AgentRunResponse(**state, changes=diff_snapshots(before, take_snapshot()))
