from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Response
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field

from create_index import create_index, get_vector_db, LOCAL_DOCS_PATH, SPLITTER_CHUNK_SIZE, SPLITTER_CHUNK_OVERLAP

from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_aws import BedrockLLM


from typing import Optional
import requests
import traceback
import io
import os
from dotenv import load_dotenv

import boto3
import botocore

os.environ["CURL_CA_BUNDLE"] = "" # Disable SSL verification

# Load environment variables
INDEX_PATH = os.getenv('INDEX_PATH', './index/')
LLM_MODEL = os.getenv('LLM_MODEL', 'anthropic.claude-v2')
RAG_PROMPT_TEMPLATE_TXT = os.getenv('RAG_PROMPT_TEMPLATE_TXT', "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use five sentences maximum and keep the answer concise.")


rag_prompt_template = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE_TXT+"""
                        Question: {question} 
                        Context: {context} 
                        Answer:""")

class RequestModel(BaseModel):
    #prompt is mandatory
    prompt: str
    session_context: Optional[str] = Field(None)
    prompt_template: Optional[str] = Field(None)


app = FastAPI(
    title = "RAG Tutorial",
    version ="1.0",
    description ="A simple API Server"
)

app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=1)




vector_db = None
is_index_creation_running = False


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def load_vector_db():
    global vector_db
    vector_db = get_vector_db()

# if index folder exist and is not empty then load the vector db
if os.path.exists(INDEX_PATH) and os.listdir(INDEX_PATH):
    load_vector_db()

# Initialize AWS Bedrock client
bedrock_client = boto3.client('bedrock-runtime')

inference_modifier = {
    "max_tokens_to_sample": 4096,
    "temperature": 0.5,
    "top_k": 250,
    "top_p": 1,
    "stop_sequences": ["\n\nHuman"],
}

llm = BedrockLLM(
    model_id=LLM_MODEL,
    client=bedrock_client,
    model_kwargs=inference_modifier,
)


#craete a endpoint to check all the environment variables
@app.get("/debug/")
async def get_env():
    data = {"INDEX_PATH":INDEX_PATH,
            "LLM_MODEL":LLM_MODEL,
            "RAG_PROMPT_TEMPLATE_TXT":RAG_PROMPT_TEMPLATE_TXT,
            "SPLITTER_CHUNK_SIZE":SPLITTER_CHUNK_SIZE,
            "SPLITTER_CHUNK_OVERLAP":SPLITTER_CHUNK_OVERLAP,
            "LOCAL_DOCS_PATH":LOCAL_DOCS_PATH}
    
    return data

def create_index_background():
    create_index()
    #check if the index is created
    if os.path.exists(INDEX_PATH) and os.listdir(INDEX_PATH):
        load_vector_db()
        print("Index created successfully")
    else:
        print("Index creation failed")

#create a endpoint to create the index
@app.post("/create_index/")
async def create_index_endpoint(background_tasks: BackgroundTasks):
    global is_index_creation_running
    if is_index_creation_running:
        return {"message":"Index creation is already running"}
    else:
        is_index_creation_running = True
        background_tasks.add_task(create_index_background)
        return {"message":"Index creation started in background"}  


@app.post("/query/")
async def generate_response(request: RequestModel):
    session_context = ""
    if request.session_context:
        session_context = request.session_context
    if len(session_context) > 0:
        local_prompt = session_context + "\n" +request.prompt
    else:
        local_prompt = request.prompt
    try:
        prompts = []
        prompts.append(local_prompt)
        #if retriever is None then generate the response without it
        if vector_db is None:
             return llm.generate(prompts)
        else:
            retriever = vector_db.as_retriever()
            docs = retriever.invoke(local_prompt)
            # Check if there are hits in the vector_db
            if not docs:
                return llm.generate(prompts)
            
            prompt_template = rag_prompt_template
            if request.prompt_template:
                prompt_template = PromptTemplate.from_template(request.prompt_template+"""
                                                                Question: {question}
                                                                Context: {context}
                                                                Answer:""")
            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt_template
                | llm
                | StrOutputParser()
            )
            
            res = rag_chain.invoke(local_prompt)
            output = {"response":res,
                    "session_context":session_context}
            return output
    except Exception as e :
        print(str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))





