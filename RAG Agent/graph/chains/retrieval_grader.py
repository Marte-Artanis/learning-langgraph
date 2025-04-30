from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

llm = ChatOpenAI(temperature=0)

class GradeDocuments(BaseModel):
    """
    Defines the structure for grading document relevance in the RAG system.
    
    This class is used to evaluate whether retrieved documents are relevant to the user's question.
    The grading is done through a binary score ('yes' or 'no') indicating document relevance.
    
    Attributes:
        binary_score (str): Indicates if documents are relevant ('yes') or not ('no')
    """
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

# Configure LLM to output structured data according to GradeDocuments schema
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# System prompt that defines the grader's role and evaluation criteria
system = """
    You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
    """

# Create a chat prompt template that combines system instructions with document and question inputs
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

# Combine the prompt template with the structured output LLM to create the final grader in a chain.
retrieval_grader = grade_prompt | structured_llm_grader
