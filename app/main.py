from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Response
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
from fastapi.responses import HTMLResponse


import traceback
import os
import json
from dotenv import load_dotenv

import boto3
from langchain_aws import ChatBedrock

load_dotenv()

# Load environment variables
LLM_MODEL = os.getenv('LLM_MODEL', 'anthropic.claude-v2')
KNOWLEDGE_BASE_ID = os.getenv('KNOWLEDGE_BASE_ID', 'your-knowledge-base-id')  # Add your KnowledgeBaseId here


# Initialize AWS Bedrock client
bedrock_client = boto3.client('bedrock-runtime')
bedrock_runtime_client = boto3.client("bedrock-agent-runtime")

# Initialize the LLM
llm = ChatBedrock(
    model_id=LLM_MODEL,
    client=bedrock_client
)

# Prompt template for extracting questions
EXTRACT_QUESTIONS_PROMPT = """
You are an assistant for processing email content. Extract all the user's questions and queries
from the following email content and return them as a JSON array without any other text in response,
not even new line characters. Only include the questions and queries 
found in the email content. Make question descriptive using the available details in email content.

Email Content: {email_content}

Questions and Queries (JSON array):
"""

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
        "LLM_MODEL": LLM_MODEL,
        "KNOWLEDGE_BASE_ID": KNOWLEDGE_BASE_ID,
    }
    return data

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("app/ui.html", "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.post("/query/")
async def generate_response(request: RequestModel):
    try:
        # Step 1: Extract questions from the email content
        extract_prompt = EXTRACT_QUESTIONS_PROMPT.format(email_content=request.email_content)
        extract_response = bedrock_client.invoke_model(
            modelId=LLM_MODEL,
            contentType='application/json',
            accept='application/json',
            body=json.dumps({
                "anthropic_version":"bedrock-2023-05-31",
                "max_tokens": 10000,  # Adjust this value as needed
                "messages": [{
                    "role": "user",
                    "content": extract_prompt
                }]
            })
        )
        extract_result = json.loads(extract_response['body'].read().decode())
        questions = json.loads(extract_result['content'][0]['text'])  # Assuming the response is a JSON array of questions

        
        responses = []
        for question in questions:
            response = bedrock_runtime_client.retrieve_and_generate(
                input={"text":question},
                retrieveAndGenerateConfiguration= {
                    "knowledgeBaseConfiguration": {
                        "knowledgeBaseId": KNOWLEDGE_BASE_ID,
                        "modelArn": LLM_MODEL
                    },
                    "type": "KNOWLEDGE_BASE"
                })
            responses.append({'question':question, 'answer':response['output']['text']})

        return {"response": responses}
    except Exception as e:
        print(str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
