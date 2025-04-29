from dotenv import load_dotenv

load_dotenv

from langchain_tavily import TavilySearch

# It convert a python function into a tool
from langchain_core.tools import StructuredTool

# It look into the state for the messages, checking the last message and see if there's any tool calls that was decided to the LLM. If there're, it's going to execute those tools.
from langgraph.prebuilt import ToolNode

from schemas import AnswerQuestion, ReviseAnswer

tavily_tool = TavilySearch(max_results=5)

#We'll take the original Tavily tool and its functionality and create 2 different tools, with diferent purpuses.
# 1 - Answer question tools, that'll be used in the initial research phase; 
# 2 - Revise answer tool, when the agent is improving it's answer based in the reflection.

# This function run queries which is going to receive as an input the search queries which is a list of strings. The kwargs are here in case the LLM is going to fail some other values. So we won't get an error in that case.
def run_queries(search_queries: list[str], **kwargs):
    return tavily_tool.batch([{'query': query} for query in search_queries])

# So we'll take the structured tool class. And it's going to have a method from function which is going to receive a function and convert it into a tool with the schema and description and all of that that we saw before.And another argument is going to be the name of the tool.
execute_tools = ToolNode(
    [
        StructuredTool.from_function(
            func=run_queries,
            name=AnswerQuestion.__name__,
            description="Run search queries for initial research"
        ),
        StructuredTool.from_function(
            func=run_queries,
            name=ReviseAnswer.__name__,
            description="Run search queries for revision"
        ),
    ]
)
