from logging import getLogger, StreamHandler, Formatter, INFO

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from hiring_agent.config.parameter_config import params_config

# load OPENAI_API_KEY from .env
load_dotenv()

# models and their settings come from params.yaml (validated on load)
agent_params = params_config.agent

# create the logger
logger = getLogger(name="Hiring Agent")
handler = StreamHandler()
logger.addHandler(handler)
logger.setLevel(INFO)
logger.propagate = False
handler.setFormatter(Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

planner_llm = ChatOpenAI(model=agent_params.planner_model)
llm = ChatOpenAI(model=agent_params.executor_model, temperature=agent_params.executor_temperature)
responder_llm = ChatOpenAI(model=agent_params.responder_model, temperature=agent_params.responder_temperature)
