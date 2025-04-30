from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.router import RouteQuery, question_router
from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState


load_dotenv()

def decide_to_generate(state):
    print("[DEBUG][decide_to_generate] state:", state)
    print("---ASSESS GRADED DOCUMENTS---")

    if state["web_search"]:
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return WEBSEARCH
    else:
        print("---DECISION: GENERATE---")
        return GENERATE


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("[DEBUG][grade_generation_grounded_in_documents_and_question] state:", state)
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    print("[DEBUG][grade_generation_grounded_in_documents_and_question] hallucination_grader score:", score)

    if hallucination_grade := score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        print("[DEBUG][grade_generation_grounded_in_documents_and_question] answer_grader score:", score)
        if answer_grade := score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"

def route_question(state: GraphState) -> str:
    print("[DEBUG][route_question] state:", state)
    print("---ROUTE QUESTION---")
    question = state["question"]
    print(f"[DEBUG][route_question] Pergunta recebida: {question}")
    source: RouteQuery = question_router.invoke({"question": question})
    print(f"[DEBUG][route_question] Fonte de dados escolhida: {source.datasource}")
    if source.datasource == WEBSEARCH:
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return WEBSEARCH
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return RETRIEVE

workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

workflow.set_conditional_entry_point(
    route_question,
    {
        WEBSEARCH: WEBSEARCH,
        RETRIEVE: RETRIEVE,
    },
)

workflow.set_entry_point(RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
    },
)

"""
Here we define the conditional edges that follow the 'generate' node in our workflow.

The function 'grade_generation_grounded_in_documents_and_question' will determine the outcome of the generation step.
It returns a string indicating the result of the reflection process:
- "not supported": The answer was not grounded in the documents, so we want to try generating a new answer.
- "useful": The answer is well-grounded and actually answers the user's question, so we can finish the workflow and return the answer.
- "not useful": The answer is grounded in the documents, but does not address the user's question, so we want to trigger a web search for more information.

We use the 'path_map' dictionary to map these string results to the actual node names in our graph:
- "not supported" → GENERATE (regenerate the answer)
- "useful" → END (finish and return the answer)
- "not useful" → WEBSEARCH (search for more information)

This approach makes the graph more explainable, as the edge labels directly reflect the reasoning process of the agent.
"""
workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported": GENERATE,
        "useful": END,
        "not useful": WEBSEARCH,
    },
)
workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")
