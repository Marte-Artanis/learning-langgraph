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











