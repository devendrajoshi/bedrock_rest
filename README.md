# RAG AWS

## Overview
RAG AWS is a project focused on implementing the Retrieval-Augmented Generation (RAG) methodology. This approach enhances the capabilities of language models by integrating retrieval mechanisms to provide more accurate and contextually relevant responses. We leverage the Large Language Model (LLM) from AWS Bedrock to achieve this. 

If you prefer to implement this without AWS Bedrock, you can follow this tutorial.

## Table of Contents
- Installation
- Running with Docker
- Testing the Endpoints
- Features
- License

## Installation
To install the project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/devendrajoshi/rag_aws.git
    ```
2. Navigate to the project directory:
    ```bash
    cd rag_aws
    ```
3. Modify the `.env` file with the following variables:
    - `INDEX_PATH`: This is the path where the index files are stored. The default value is `./index/`.
    - `EMBEDDING_MODEL_NAME`: This is the name of the model used for generating embeddings. The default model is `sentence-transformers/all-mpnet-base-v2`.
    - `LOCAL_DOCS_PATH`: This is the directory where the application looks for PDF files to create embeddings and store them in a vector database. The default path is `./localdocs/`. Users should mount their folder containing their PDF files to the FastAPI container.
    - `SPLITTER_CHUNK_SIZE`: This is the size of the chunks that the document splitter will create. The default chunk size is `1000`.
    - `SPLITTER_CHUNK_OVERLAP`: This is the overlap between chunks that the document splitter will create. The default overlap is `200`.
    - `LLM_MODEL`: This is the name of the LLM model used. The default model is `llama3.1`.
    - `LLM_HOST`: This is the host address for the LLM service. The default host is `127.0.0.1`.
    - `LLM_PORT`: This is the port number for the LLM service. The default port is `11434`.
    - `RAG_PROMPT_TEMPLATE_TXT`: This is the prompt template text for the RAG model. The default text is "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use five sentences maximum and keep the answer concise."
    - AWS connection variables:
        - `AWS_DEFAULT_REGION`
        - `AWS_ACCESS_KEY_ID`
        - `AWS_SECRET_ACCESS_KEY`
        - `AWS_SESSION_TOKEN`

These variables are used by the `boto3` library. Note that this method of putting access keys and secret keys in the `.env` file is only for demo purposes and is not recommended for production use. For more information, refer to the Boto3 documentation. **Never push `.env` files with your keys to any repository.**

## Running with Docker
We recommend using Docker to run this project. The necessary `Dockerfile` and `docker-compose.yml` are included in the source code.

### Commands to Run
1. Build and run the Docker containers:
    ```bash
    docker-compose up --build
    ```

2. Access the application at `http://localhost:9002`.

### Docker Installation
If you don't have Docker installed, you can follow the instructions on the Docker website to install it on your system.

**Note:** You may also run this project without Docker if you prefer. In that case, we expect you to know how to run a FastAPI application.

## Testing the Endpoints
You can test the endpoints using `curl` commands. Here are the endpoints to test:

1. **Check Server Configuration**
    ```bash
    curl http://127.0.0.1:9002/debug
    ```
    This endpoint shows basic variables to test if the server is up with the correct configurations.

2. **Create Index**
    ```bash
    curl http://127.0.0.1:9002/create_index
    ```
    This endpoint initiates the index creation process, which may take time based on the number of PDF files in the `localdocs` folder. Ensure that when updating the `.env` file, you point `LOCAL_DOCS_PATH` to a folder that contains the PDF files to be used for RAG. The current code only works for PDF files but can be extended for other file types.

3. **Query LLM with RAG**
    ```bash
    curl -X POST http://127.0.0.1:9002/query -H "Content-Type: application/json" -d '{"prompt":"What are the key features of the RAG methodology?"}'
    ```
    This is the main API where you query the LLM with RAG. It takes a JSON payload with a `prompt` field. Replace the example prompt with your own query.

## Features
- Implementation of RAG methodology
- Integration with AWS Bedrock LLM
- Enhanced retrieval mechanisms for improved response accuracy
- Tested with the `anthropic.claude-v2` model

## License
This project is licensed under the [License Name]. See the LICENSE file for more details.
