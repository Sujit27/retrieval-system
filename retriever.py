import os
import config

import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import tempfile


class Retriever:
    '''
    '''
    def __init__(self,embedding_model="all-MiniLM-L6-v2",llm_model='open-ai'):
        
        self.text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200,separator = " ")
        self.documents = None

        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)

        if llm_model == 'open-ai':
            self.llm_model = ChatOpenAI(temperature=0.0,model_name='gpt-3.5-turbo')

        self.retriever_chain = None

    