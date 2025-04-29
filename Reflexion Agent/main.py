from dotenv import load_dotenv
import os
load_dotenv()

if __name__ == '__main__':
    print('Hello, LangGraph')
    print('LANGCHAIN_API_KEY:', os.getenv('LANGCHAIN_API_KEY'))
    print('LANGCHAIN_TRACING_V2:', os.getenv('LANGCHAIN_TRACING_V2'))
    print('LANGCHAIN_PROJECT:', os.getenv('LANGCHAIN_PROJECT'))
    print('TAVILY_API_KEY:', os.getenv('TAVILY_API_KEY'))