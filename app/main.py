from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Response
from pydantic import BaseModel, Field

from typing import Optional, List
import requests
import traceback
import io
import os
import json
from dotenv import load_dotenv

import boto3
import botocore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_aws import BedrockLLM

os.environ["CURL_CA_BUNDLE"] = ""  # Disable SSL verification

# Load environment variables
INDEX_PATH = os.getenv('INDEX_PATH', './index/')
LLM_MODEL = os.getenv('LLM_MODEL', 'anthropic.claude-v2')
KNOWLEDGE_BASE_ID = os.getenv('KNOWLEDGE_BASE_ID', 'your-knowledge-base-id')  # Add your KnowledgeBaseId here

# Initialize AWS Bedrock client
bedrock_client = boto3.client('bedrock-runtime')

# Initialize the LLM
llm = BedrockLLM(
    model_id=LLM_MODEL,
    client=bedrock_client
)

# Prompt template for extracting questions
EXTRACT_QUESTIONS_PROMPT = """
You are an assistant for processing email content. Extract all the questions and queries from the following email content and return them as a JSON array.

Email Content: {email_content}

Questions and Queries (JSON array):
"""

# System prompt for generating responses
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

# Chat prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create the document combination chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)

class RequestModel(BaseModel):
    email_content: str

app = FastAPI(
    title="Email Query Processor",
    version="1.0",
    description="A simple API Server for processing email content"
)

app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=1)

@app.get("/debug/")
async def get_env():
    data = {
        "INDEX_PATH": INDEX_PATH,
        "LLM_MODEL": LLM_MODEL,
        "KNOWLEDGE_BASE_ID": KNOWLEDGE_BASE_ID,
    }
    return data

@app.post("/query/")
async def generate_response(request: RequestModel):
    try:
        # Step 1: Extract questions from the email content
        extract_prompt = EXTRACT_QUESTIONS_PROMPT.format(email_content=request.email_content)
        extract_response = bedrock_client.invoke_endpoint(
            EndpointName='your-endpoint-name',  # Replace with your endpoint name
            ContentType='application/json',
            Body=json.dumps({
                "prompt": extract_prompt,
                "model_id": LLM_MODEL
            })
        )
        extract_result = json.loads(extract_response['Body'].read().decode())
        questions = extract_result  # Assuming the response is a JSON array of questions

        # Step 2: Create the retrieval chain
        retriever = create_retrieval_chain(bedrock_client, KNOWLEDGE_BASE_ID)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # Step 3: Generate responses for each question
        responses = []
        for question in questions:
            response = rag_chain.invoke({"input": question})
            responses.append(response.get('response', ''))

        # Step 4: Combine responses into a cohesive reply
        cohesive_response = "\n\n".join(responses)
        return {"response": cohesive_response}
    except Exception as e:
        print(str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
