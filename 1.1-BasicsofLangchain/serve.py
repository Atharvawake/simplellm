from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langserve import add_routes

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq model
model = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

# Define prompt template
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages([
    {"role": "system", "content": system_template},
    {"role": "user", "content": "{text}"}
])

# Define output parser
parser = StrOutputParser()

# Create chain
chain = prompt_template | model | parser

# Initialize FastAPI app
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API server using Langchain runable interface"
)

# Add routes for the chain
add_routes(
    app,
    chain,
    path="/chain"
)

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
