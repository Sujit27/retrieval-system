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
from langchain import HuggingFaceHub
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain

warnings.filterwarnings("ignore")

filePath1 = 'input_samples/Dynamic_density.txt'
filePath2 = 'input_samples/state_of_the_union.txt'
os.environ["OPENAI_API_KEY"] = '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'
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'
os.environ["TOKENIZERS_PARALLELISM"] = 'false'

def main():
    
    loader1 = TextLoader(filePath1, encoding='utf8')
    loader2 = TextLoader(filePath2, encoding='utf8')

    ### For multiple documents 
    loaders = [loader1,loader2]
    documents = []
    for loader in loaders:
        documents.extend(loader.load())
    # print("The documents are loaded")
    

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200,separator = " ")
    texts = text_splitter.split_documents(documents)
    # print("Text split completed")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # print("Embedding model loaded")
    db = Chroma.from_documents(texts, embeddings)
    # print("Embeddings generated")

    retriever = db.as_retriever(search_type="similarity",search_kwargs={"k": 3})

    llm=OpenAI()
    # # llm = HuggingFaceHub(repo_id="bigscience/bloom-7b1", model_kwargs={"temperature":0, "max_length":512})

    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)

    while True:
        query = input("Enter a query, or press q/Q to exit:\n")
        if query.lower() == 'q':
            break
        else:
            print("Running the query...")
            print(qa.run(query))
            
            # print("Getting relevant text from documents")
            # docs = retriever.get_relevant_documents(query)
            # for doc in docs:
            #     print(doc)
            #     print("\n\n")


if __name__ == '__main__':
    main()


