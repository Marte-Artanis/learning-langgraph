from dotenv import load_dotenv

load_dotenv()

from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.generation import generation_chain
from graph.chains.hallucination_grader import hallucination_grader, GradeHallucinations
from graph.chains.router import question_router, RouteQuery

from ingestion import retriever

def test_retrieval_grader_anwer_yes() -> None:
    question = 'agent memory'
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {'question': question, 'document': doc_txt}
    )

    assert res.binary_score == 'yes'

def test_retrieval_grader_answer_no() -> None:
    question = 'agent memory'
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_count

    res: GradeDocuments = retrieval_grader.invoke(
        {'question': 'How to make pizza', 'document': doc_txt}
    )
    
    assert res.binary_score == 'no'

def test_generation_chain() -> None:
    question = 'agent memory'
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({'context': docs, 'question': question})
    print(generation)

def test_halluctination_grade_answear_yes() -> None:
    """
    This test checks if the hallucination grader correctly identifies that the generated answer
    is grounded in the retrieved documents.

    Steps:
    1. It defines a question ("agent memory") and retrieves relevant documents using the retriever.
    2. It generates an answer using the generation chain, based on the retrieved documents and the question.
    3. It passes both the documents and the generated answer to the hallucination grader.
    4. The hallucination grader should return a positive binary score (True/Yes), indicating that
       the answer is indeed grounded in the provided documents and is not a hallucination.

    The test passes if the hallucination grader confirms the answer is grounded (not a hallucination).
    """
    question = 'agent memory'
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({'context': docs, 'question': question})
    
    res: GradeHallucinations = hallucination_grader.invoke(
        {'documents': docs, 'generation': generation}
    )

    assert res.binary_score

def test_halluctination_grade_answear_no() -> None: 
    question = 'agent memory'
    docs = retriever.invoke(question)
    
    res: GradeHallucinations = hallucination_grader.invoke(
        {
            'documents': docs, 
            'generation': 'In order to make pizza, we need to first start with the dough.'}
    )

    assert not res.binary_score

def test_router_to_vectorstore() -> None:
    question = 'agent memory'
    
    res: RouteQuery = question_router.invoke(
        {
            'question': question, 
        }
    )

    assert res.datasource == 'vectorstore'

def test_router_to_websearch() -> None:
    question = 'How to make pizza'
    
    res: RouteQuery = question_router.invoke(
        {
            'question': question, 
        }
    )

    assert res.datasource == 'websearch'
