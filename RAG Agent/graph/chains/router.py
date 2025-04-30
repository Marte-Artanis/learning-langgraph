# The literal type like I was a couple of months ago, then it provides a way to specify that a variable can only take one of predefined set of values. So this is very useful for validation and type checking.
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

class RouteQuery(BaseModel):
    # Route a user query to the most relevant datasource.

    # datasourse: holds the values of vectorsore - in case we want the look for the answer in the vectorstore - or it'll going to hold websearch, in case we want to look for the answer in the web.

    # The ellisis means thatthis field is required once we instantiate an object of this class.
    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )


llm = ChatOpenAI(temperature=0)
structured_llm_router = llm.with_structured_output(RouteQuery)

system = """
    You are an expert at routing a user question to a vectorstore or web search.
    The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
    Use the vectorstore for questions on these topics. For all else, use web-search.
    """

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router
