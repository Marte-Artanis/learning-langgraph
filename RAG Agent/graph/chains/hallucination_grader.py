"""
In this step, we are adding an automatic reflection layer to the agent, inspired by the Self-RAG approach.

After the agent generates an answer, it doesn't finish immediately. Instead, the answer goes through two graders:
- One that checks if the answer is grounded in the retrieved documents (to avoid hallucinations).
- Another that checks if the answer actually addresses the user's original question.

These graders will decide the agent's next step:
- If the answer is correct and well-grounded, the flow ends.
- If not, the agent can try to generate a new answer or perform another search.

We chose to use conditional branching in the graph, instead of just a single node, to flexibly decide whether the agent should finish, try again, or look for more data.

This makes the agent smarter and more aligned with the idea of automatic reflection, allowing it to learn from its own mistakes and improve the answers for the user.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

llm = ChatOpenAI(temperature=0)


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

structured_llm_grader = llm.with_structured_output(GradeHallucinations)

system = """
    You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'yes' means that the answer is grounded in / supported by the set of facts.
     """

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader
