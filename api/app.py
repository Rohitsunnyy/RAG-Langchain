from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import OpenAI
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv

load_dotenv()

os.environ['OPEN_API_KEY'] = os.getenv("OPENAI_API_KEY")

app = FastAPI(
    title="Langchain Server",
    version='1.0',
    description="A simple API Server"
)

openai_llm = OpenAI(model="gpt-3.5-turbo")

add_routes(
    app,
    openai_llm,
    path="/openai"
)

model = OpenAI(model="gpt-3.5-turbo")
ollama_llm = Ollama(model='llama3')

prompt1 = ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")
prompt2 = ChatPromptTemplate.from_template("Write me a poem about {topic} with 100 words")

add_routes(
    app,
    prompt1 | model,
    path="/essay"
)

add_routes(
    app,
    prompt2 | ollama_llm,
    path="/poem"
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8002)
