from typing import List, TypedDict, Annotated
from langchain.schema import Document
from langgraph.channels import BinaryOperatorAggregate
import operator


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: Annotated[str, BinaryOperatorAggregate(str, operator.add)]
    generation: Annotated[str, BinaryOperatorAggregate(str, operator.add)]
    web_search: Annotated[bool, BinaryOperatorAggregate(bool, operator.or_)]
    documents: Annotated[List[Document], BinaryOperatorAggregate(List[Document], operator.add)]
