import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.vectorstores.faiss import FAISS
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import base64
from tempfile import NamedTemporaryFile
import streamlit as st
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, Table, MetaData, select, text
from sqlalchemy import inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String
from sqlalchemy.sql import exists
from sqlalchemy import text
import json
import yaml

def read_config_file(config_file):
    with open(config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

Base = declarative_base()

def get_connection_string():
    config = read_config_file("./config.yml")

    return PGVector.connection_string_from_db_params(
        driver=config["DB"]["DRIVER"],
        host=config["DB"]["HOST"],
        port=config["DB"]["PORT"],
        database=config["DB"]["DATABASE"],
        user=config["DB"]["USER"],
        password=config["DB"]["PASSWORD"],
    )

class ProcessedDocument(Base):
    __tablename__ = 'processed_documents'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)

class VectorDB_PG(PGVector):
    def __init__(self, connection_string=get_connection_string(), embedding_function=None, collection_name="protocols", openai_api_key=None):
        self.connection_string = connection_string
        self.openai_api_key = openai_api_key
        self.embedding_function = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        self.collection_name = collection_name
        self.engine = create_engine(self.connection_string)
        self.metadata = MetaData()

        super().__init__(
            connection_string=self.connection_string, 
            embedding_function=self.embedding_function, 
            collection_name=self.collection_name)
        
        Base.metadata.create_all(self.engine)

    def document_exists(self, document_name):
        with Session(self.engine) as session:
            return session.query(exists().where(ProcessedDocument.name == document_name)).scalar()
            
    def add_processed_document(self, document_name, overwrite=None):
        with Session(self.engine) as session:
            processed_document = ProcessedDocument(name=document_name)
            session.add(processed_document)
            session.commit()

    def create_tables(self):
        Base.metadata.create_all(self.engine)

    def compose_prompt(self, context, query):
        return "Context: " + context + "\n\n" + "Question: " + query + "\n\n"
    
    def get_compose_prompt(self, query, document_name=None):
        if document_name:
            filter = {"source":f"protocols/{document_name}"}
            results = self.similarity_search_with_score(query, filter=filter, k=5)
            context = " ".join([c[0].page_content + str("Chunk Number: ",i) for i,c in enumerate(results)])
            return self.compose_prompt(context, query)
        else:
            results = self.similarity_search_with_score(query,k=5)
            context = " ".join([c[0].page_content for c in results])
            return self.compose_prompt(context, query)
    
    def get_document_names(self):
        with Session(self.engine) as session:
            return [name[0] for name in session.query(ProcessedDocument.name).all()]
        
    def create_extension(self):
        with Session(self.engine) as session:
            session.execute(text("CREATE EXTENSION vector;"))
            session.commit()

class VectorDB_FAISS(FAISS):
    def __init__(self, openai_api_key=None):
        self.openai_api_key = openai_api_key
        self.embedding_function = OpenAIEmbeddings(openai_api_key=self.openai_api_key)

        super().__init__(
            embedding_function=self.embedding_function)

    def create_tables(self):
        Base.metadata.create_all(self.engine)

    def compose_prompt(self, context, query):
        return "Context: " + context + "\n\n" + "Question: " + query + "\n\n"
    
    def get_compose_prompt(self, query, document_name=None):
        if document_name:
            filter = {"source":f"protocols/{document_name}"}
            results = self.similarity_search_with_score(query, filter=filter, k=5)
            context = " ".join([c[0].page_content + str("Chunk Number: ",i) for i,c in enumerate(results)])
            return self.compose_prompt(context, query)
        else:
            results = self.similarity_search_with_score(query,k=5)
            context = " ".join([c[0].page_content for c in results])
            return self.compose_prompt(context, query)
    
    def get_document_names(self):
        with Session(self.engine) as session:
            return [name[0] for name in session.query(ProcessedDocument.name).all()]
        
    def create_extension(self):
        with Session(self.engine) as session:
            session.execute(text("CREATE EXTENSION vector;"))
            session.commit()

class ProcessDocument:
    def __init__(self, path):
        self.path = path
        self.documents = self.load_and_chunk()

    def load_and_chunk(self, no_chunk=False):
        loader = PyPDFLoader(self.path)
        raw_documents = loader.load()
        
        if no_chunk:
            return raw_documents
        else:
            text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2048,
            chunk_overlap=0,
        )
            chunks = text_splitter.split_documents(raw_documents)
        
            return chunks

class SaveFileToDisk:
    def __init__(self, uploaded_file, vector_db, directory="protocols"):
        self.uploaded_file = uploaded_file
        self.vector_db = vector_db
        self.directory = directory
        self.file_path = self.save_uploaded_file()  
        
    def save_uploaded_file(self):
        os.makedirs(self.directory, exist_ok=True)
        file_path = os.path.join(self.directory, self.uploaded_file.name)
        
        if not self.vector_db.document_exists(self.uploaded_file.name):
            with open(file_path, "wb") as f:
                f.write(self.uploaded_file.getbuffer())

        return file_path 

class DisplayDocument:
    def __init__(self, saved_path):
        self.saved_path = saved_path

    def displayPDF(self):
        with open(self.saved_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="600" height="700" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

class PromptLoader:
    def __init__(self, path):
        self.path = path
        self.prompt = self.load_prompt()

    def load_prompt(self):
        with open(f"prompts/{self.path}") as p:
            return json.load(p)