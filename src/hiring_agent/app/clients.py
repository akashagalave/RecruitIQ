from logging import getLogger, StreamHandler, Formatter, INFO

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()

PLANNER_MODEL = "gpt-4o-mini"  
EXECUTOR_MODEL = "gpt-4.1-mini"  


logger = getLogger(name="Hiring Agent")
handler = StreamHandler()
logger.addHandler(handler)
logger.setLevel(INFO)
logger.propagate = False
handler.setFormatter(Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

planner_llm = ChatOpenAI(model=PLANNER_MODEL, temperature=0)
llm = ChatOpenAI(model=EXECUTOR_MODEL, temperature=0)


