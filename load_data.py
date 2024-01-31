# Import libraries
import io
import requests
import traceback
from typing import List
import json
import boto3
import psycopg
import numpy as np
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.llms import Bedrock
from langchain.prompts import PromptTemplate
from langchain.schema.retriever import BaseRetriever
from langchain.schema import Document, HumanMessage
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.chains import RetrievalQA
from pgvector.psycopg import register_vector


def similarity_search(query):
    query_embedding = bedrock_embeddings.embed_query(query)
    dbconn = psycopg.connect(conninfo=dbconnstring, connect_timeout=10)
    register_vector(dbconn)
    result = dbconn.execute("""SELECT document FROM qa_rag_rls ORDER BY
                            embedding <=> %s limit 3;""",(np.array(query_embedding),)).fetchall()
    dbconn.close()
    return [ i[0] for i in result ]


def get_pdf_text(pdf_docs):
    text = ""
    pdf_reader = PdfReader(pdf_docs)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
     )

    chunks = text_splitter.split_text(text)
    return chunks


def create_table():
    dbconn = psycopg.connect(conninfo=dbconnstring,autocommit=True)
    dbconn.execute("""create table if not exists load_vector(n serial, document text, embedding vector(1536);""")
    dbconn.close()

def store_vector(chunks,to_repeat):

    dbconn = psycopg.connect(conninfo=dbconnstring,autocommit=True)
    register_vector(dbconn)
    ee = bedrock_embeddings.embed_documents(chunks)

    for i in range(to_repeat):
        for idx, x in enumerate(chunks):
             dbconn.execute("""INSERT INTO load_vector (document, embedding)
                          VALUES( %s, %s);""", ( x, ee[idx]))
    dbconn.close()


def main():

    create_table()

    #pdf_list = ['https://d0.awsstatic.com/whitepapers/Migration/amazon-aurora-migration-handbook.pdf']
    to_repeat = 1000
    pdf_list = None
    with open("pdf_files.lst") as fp:
        pdf_list = fp.readlines()

    for pdf in pdf_list:
        filename = pdf.strip()
        print("Processing the file {}".format(filename))
        response = requests.get(filename)
        pdf_data = response.content
        raw_text = get_pdf_text(io.BytesIO(pdf_data))
        text_chunks = get_text_chunks(raw_text)
        store_vector(text_chunks,to_repeat)

if __name__ == '__main__':

    BEDROCK_CLIENT = boto3.client("bedrock-runtime",region_name="us-east-1")
    bedrock_embeddings = BedrockEmbeddings(model_id= "amazon.titan-embed-text-v1", client=BEDROCK_CLIENT)
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId='apgpg-pgvector-secret')
    database_secrets = json.loads(response['SecretString'])
    dbhost = database_secrets['host']
    dbport = database_secrets['port']
    dbuser = database_secrets['username']
    dbpass = database_secrets['password']
    dbconnstring = "postgresql://{}:{}@{}:{}".format(dbuser,dbpass,dbhost,dbport)

    main()
