

from fastapi import FastAPI

from hiring_agent.app import workspace

from hiring_agent.api.routers import agent, health
from hiring_agent.api.routers import workspace as workspace_router

workspace.use_workspace(workspace.APP_WORKSPACE_DIR)
if not workspace.APP_WORKSPACE_DIR.exists():
    workspace.create_empty_workspace()

app = FastAPI(
    title="Hiring Assistant API",
    description="Runs the hiring agent and serves its workspace to the Streamlit frontend.",
)

app.include_router(agent.router)
app.include_router(workspace_router.router)
app.include_router(health.router)
