import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
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

Base = declarative_base()

CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver=os.environ.get("PGVECTOR_DRIVER", "psycopg2"),
    host=os.environ.get("PGVECTOR_HOST", "vector-db-iqvia.c01dryusnrkr.ap-south-1.rds.amazonaws.com"),
    port=int(os.environ.get("PGVECTOR_PORT", "5432")),
    database=os.environ.get("PGVECTOR_DATABASE", "postgres"),
    user=os.environ.get("PGVECTOR_USER", "postgres"),
    password=os.environ.get("PGVECTOR_PASSWORD", "Anshmania2020"),
)
class ProcessedDocument(Base):
    __tablename__ = 'processed_documents'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)

class VectorDB(PGVector):
    def __init__(self, connection_string=CONNECTION_STRING, embedding_function=None, collection_name="protocols", openai_api_key=None):
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
    
    def get_context(self, query, document_name=None):
        if document_name:
            filter = {"source":f"protocols/{document_name}"}
            results = self.similarity_search_with_score(query, filter=filter, k=5)
            return " ".join([c[0].page_content + str("Chunk Number: ",i) for i,c in enumerate(results)])
        else:
            results = self.similarity_search_with_score(query,k=5)
            return " ".join([c[0].page_content for c in results])
    
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
            return text_splitter.split_documents(raw_documents)
        
    def analyze_chunks(self):
        pass
    
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