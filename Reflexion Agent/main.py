from dotenv import load_dotenv
from typing import List

load_dotenv()

from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import END, MessageGraph

from chains import first_responder, revisor
from tool_executor import execute_tools

MAX_ITERATIONS = 2

# Initialize the message graph builder
builder = MessageGraph()

# Add nodes to the graph:
# 1. "draft" - generates initial response
# 2. "execute_tools" - performs research using Tavily
# 3. "revise" - improves the answer based on research
builder.add_node("draft", first_responder)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revise", revisor)

# Define the flow of the graph:
# draft -> execute_tools -> revise
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")

# Event loop function to control the research iterations
# It counts how many times we've used tools and stops after MAX_ITERATIONS
def event_loop(state: List[BaseMessage]) -> str:
    # Count how many times we've used tools in this iteration
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
    num_iterations = count_tool_visits
    
    # If we've exceeded max iterations, end the process
    if num_iterations > MAX_ITERATIONS:
        return END
    
    # Otherwise, continue with more research
    return "execute_tools"

# Add conditional edge from revise node based on event_loop function
# This creates a loop: revise -> execute_tools -> revise (until max iterations)
builder.add_conditional_edges("revise", event_loop)

# Set the entry point of our graph
builder.set_entry_point("draft")

# Compile the graph into an executable
graph = builder.compile()

# Run the graph with our research topic
res = graph.invoke(
    "Write about AI-Powered SOC / autonomous soc  problem domain, list startups that do that and raised capital."
)

# Print the final answer and full response history
print(res[-1].tool_calls[0]["args"]["answer"])
print(res)
