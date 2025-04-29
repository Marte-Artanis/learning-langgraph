from typing import List

from pydantic import BaseModel, Field

# Now this is another cool trick where we're actually prompting LLM through the description of the class's.
class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous")


class AnswerQuestion(BaseModel):
    answer: str = Field(description="~250 word detailed answer to the question.")
    reflection: Reflection = Field(description="Your reflection on the initial answer.")
    search_queries: List[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )

# Forcing citation in the model encourages grounded responses
class ReviseAnswer(AnswerQuestion):
    references: List[str] = Field(
        description="Citations motivating your updated answer."
    )
