import datetime
from dotenv import load_dotenv

load_dotenv()

# Both output parsers take back the response we get the LLM with the function calling invocation and transform it into JSON to a dictionary or a Pydantic object.
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from schemas import AnswerQuestion

llm = ChatOpenAI(model="gpt-4o")

# We're going to create two output parsers.
# The first one is going to be a JSON output tools parser, which is simply going to return us the function call we got back from the LM and transform it into a dictionary. And the second output parser is going to be a pedantic tools output parser, which is going to take the response from the LM.
parser = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

# The input for the agent will be the topic we want to talk about and the agent will write the first response. In the answear of the agent we need the content, the first draft of the article, a critique and some search term to improve the article.
# We'll use the message graph. So the state of the graph nodes will be changing upon every nod, becoming a list of messages.
# That's a chat prompt template from the messages. so the messages will me here and all the history so far.
# This prompt template will be used to our revisor node - that will take the information and rewrite the article.

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", 
         """
        You are a expert researcher.
        Current time: {time}

        1. {first_instruction}
        2. Reflect and critique your answer. Be severe to maximize improvement.
        3. Write 1-3 search queries to research the topic. Your response MUST include a 'search_queries' field with a list of strings.

        """
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(time=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# The parcial method populate the placeholders, because when we invoke the prompt template, we want plug in the current time and the lambda function do that.

# But now we want to ensure that the output we get from the LM, then it is in a structured format.
# So we want the format to be with a response field that is having the original essay.
# We want the critique field, which is having the critique for that essay, and we want a search field,
# which will be a list of values that we should search for.

first_responder_prompt_template = actor_prompt_template.partial(first_instruction="Provide a detailed ~250 word answer.")

# Let's now create the first responder chain which is going to take our prompt template.
# And it's going to pipe it into the LLM GPT.
# But not before we bind the answer question object as a tool for the function calling and by providing tool choice equals answer question.
# This will force the LLM to always use the answer question tool, thus grounding the response to the object that we want to receive.
# And this is a cool technique where the grounding of the LM also comes from the Pydantic object we created.
# So this way we are going to make the LLM give us exactly the response that we want.

first_responder = first_responder_prompt_template | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")

if __name__ == '__main__':
    human_message = HumanMessage(
        content="Write about AI-Powered SOC/ automated SOC problems domain"
    )

    chain = first_responder_prompt_template | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion") | parser_pydantic

    res = chain.invoke({"messages": [human_message]})
    print(res)

