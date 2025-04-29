# Lesson n . 13: Build Graph

from dotenv import load_dotenv
load_dotenv()

from typing import Sequence, List
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph
import os

from chains import generate_chain, reflection_chain

REFLECT = 'reflect'
GENERATE = 'generate'

# __start__ -> generate <-> reflect -> __end__

# MessageGraph: it's a sequence of messages, and the input is a list of messages, and the output is going to be one or a couple of message.

# FLOW EXPLANATION:
# The 'state' (or 'messages') is a list representing the conversation history.
# Each node in the graph receives the current state (all accumulated messages so far),
# generates a new message (or messages), and appends it to the state.
# This updated state is then passed to the next node.
# The process continues, growing the state at each step, until the graph ends.
#
# Example:
# 1. Start: [HumanMessage("Write a tweet about AI.")]
# 2. generation_node: receives state, generates tweet -> [HumanMessage(...), AIMessage(...)]
# 3. reflection_node: receives state, generates critique -> [HumanMessage(...), AIMessage(...), HumanMessage(...)]
# 4. Next node: receives updated state, and so on.
#
# The MessageGraph manages this state automatically, always passing the full message history to each node.
# Each function below is responsible for receiving the current state and returning new messages to be appended.

# The node will run the generation chain, invoking with all the state we have already accumulated.
# The node takes the tweet and revises according to the feedback we get from the reflection node.
# We'll take the return value and append to the state.
def generation_node(state: Sequence[BaseMessage]):
    return generate_chain.invoke({'messages': state})

# The node will run the reflection chain, invoking with all the state we have already accumulated.
# The response we get back from the LLM (usually with role AI) will be changed to a HumanMessage.
# This is done to trick the LLM into thinking a human is sending this message, enabling a conversation loop.
def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    res = reflection_chain.invoke ({'messages': messages})
    return [HumanMessage(content=res.content)]


# Now, we inicialize the MessageGraph.
builder = MessageGraph()

# We add the nodes to the graph.
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)

# We set the entry point of the graph.
builder.set_entry_point(GENERATE)

# Let's say we executed the generation node and we revised the tweet once. Now we want to implement the logic that'd say that the logic that the tweet is good enough and we can finish or that we need the reflections step. For that we'll use the function should continue, which will receive the state - messages - and decide which node should we go to - the end node or the reflection node. The return value of this function is a string, and it's the key of the node that we want to go to. That's call the conditional edge.

# Here we'll say that if the length of the state is greater than 6, we'll return the end node. Otherwise, we'll return the reflection node. The LLM won't decide the node, we'll decide the node.

# Now we want to tell the LangGraph that after we finish executing the generate node and we generated the tweet, we want now to add a conditional edge that the function should_continue will determine which node to go to next - the end node or the reflection node. 
def should_continue(state: List[BaseMessage]):
    if len(state) > 6:
        return END
    return REFLECT

builder.add_conditional_edges(GENERATE, should_continue)

# After the feedback, we want to revise the tweet. So we'll add an edge between the reflection node and the generate node.
builder.add_edge(REFLECT, GENERATE)

# Now we build the graph.
graph = builder.compile()

# we need a way to visualize the graph.
print(graph.get_graph().draw_mermaid())
graph.get_graph().print_ascii()

if __name__ == '__main__':
    inputs = HumanMessage(content="""
    Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post
    """)

    response = graph.invoke(inputs)
