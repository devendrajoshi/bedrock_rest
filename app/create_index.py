import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import boto3
from datetime import datetime
import requests


load_dotenv()

# Load environment variables
INDEX_PATH = os.getenv('INDEX_PATH', './index/')
#make sure INDEX_PATH ends with '/'
if not INDEX_PATH.endswith('/'):
    INDEX_PATH = INDEX_PATH + '/'
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', 'sentence-transformers/all-mpnet-base-v2')
LOCAL_DOCS_PATH = os.getenv('LOCAL_DOCS_PATH', './localdocs/')
#make sure LOCAL_DOCS_PATH ends with '/'
if not LOCAL_DOCS_PATH.endswith('/'):
    LOCAL_DOCS_PATH = LOCAL_DOCS_PATH + '/'
    
SPLITTER_CHUNK_SIZE = int(os.getenv('SPLITTER_CHUNK_SIZE', 1000))
SPLITTER_CHUNK_OVERLAP = int(os.getenv('SPLITTER_CHUNK_OVERLAP', 200))

splitter = RecursiveCharacterTextSplitter(chunk_size=SPLITTER_CHUNK_SIZE, chunk_overlap=SPLITTER_CHUNK_OVERLAP)


def get_vector_db():
    requests.packages.urllib3.disable_warnings()
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_db = Chroma(
        collection_name="local_collection",
        embedding_function=embeddings,
        persist_directory=INDEX_PATH+"embeddings_db",  # Where to save data locally, remove if not neccesary
    )
    return vector_db


def create_index():
    current_time = datetime.now()
    vector_db = get_vector_db()
    #empty the vector_db before adding new documents
    vector_db.reset_collection()
    doc_cnt = 0
    #if the LOCAL_DOCS_PATH is local file path
    if os.path.exists(LOCAL_DOCS_PATH) and os.listdir(LOCAL_DOCS_PATH):
        doc_cnt = read_local_files(vector_db)
    
    elif LOCAL_DOCS_PATH.startswith('s3://'):
        doc_cnt = read_s3_files(vector_db)

    else:
        print("Invalid LOCAL_DOCS_PATH or empty directory: ", LOCAL_DOCS_PATH)

    print(f"Indexed {doc_cnt} documents in {datetime.now() - current_time}")

def process_pdf(file, source, doc_cnt):
    global splitter
    documents = []
    id_cnt = doc_cnt
    reader = PyPDF2.PdfReader(file)
    text = ''
    page_cnt = 0
    for page in range(len(reader.pages)):
        text = reader.pages[page].extract_text()
        page_cnt += 1
        for chunk in splitter.split_text(text):
            id_cnt += 1
            doc = Document(
                page_content=chunk,
                metadata={"source": source, "page": page_cnt},
                id=id_cnt,
            )
            documents.append(doc)
    print(f"Read {page_cnt} pages")
    return documents

def read_local_files(vector_db):
    doc_cnt = 0
    for filename in os.listdir(LOCAL_DOCS_PATH):
        if filename.endswith('.pdf'):
            with open(os.path.join(LOCAL_DOCS_PATH, filename), 'rb') as file:
                print(f"Reading {filename}")
                docs = process_pdf(file, filename, doc_cnt)
                doc_cnt += len(docs)
                vector_db.add_documents(documents=docs)
    return doc_cnt

def read_s3_files(vector_db):
    doc_cnt = 0
    s3 = boto3.client('s3')
    # Extract bucket name and s3_docs_path from LOCAL_DOCS_PATH
    path_parts = LOCAL_DOCS_PATH.replace("s3://", "").split("/", 1)
    bucket_name = path_parts[0]
    s3_docs_path = path_parts[1] if len(path_parts) > 1 else ""
    
    # List all objects in the specified S3 path
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_docs_path)
    
    for obj in response.get('Contents', []):
        key = obj['Key']
        if key.endswith('.pdf'):
            print(f"Reading {key}")
            
            # Download the file from S3
            s3.download_file(bucket_name, key, '/tmp/temp.pdf')
            
            with open('/tmp/temp.pdf', 'rb') as file:
                docs = process_pdf(file, key, doc_cnt)
                doc_cnt += len(docs)
                vector_db.add_documents(documents=docs)
    
    return doc_cnt