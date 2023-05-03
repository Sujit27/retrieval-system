import os
import logging
import warnings

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from abc import ABC, abstractmethod
from typing import List
from langchain.schema import Document

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings 
from langchain.vectorstores import Chroma

warnings.filterwarnings("ignore")

filePath = 'input_samples/state_of_the_union.txt'
os.environ["OPENAI_API_KEY"] = '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'
os.environ["TOKENIZERS_PARALLELISM"] = 'false'

def main():
    print('start')
    loader = TextLoader(filePath, encoding='utf8')
    documents = loader.load()
    print("The documents are loaded")
    

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    print("Text split completed")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("Embedding model loaded")
    db = Chroma.from_documents(texts, embeddings)
    print("Embeddings generated")

    retriever = db.as_retriever()

    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)

    query = "What did the president say about children"
    print("Running the query...")
    print(qa.run(query))


if __name__ == '__main__':
    main()


