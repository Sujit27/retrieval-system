import os
import config
from retriever import Retriever

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



os.environ["OPENAI_API_KEY"] = config.open_ai_api_key

# uploaded_files = st.sidebar.file_uploader("upload", accept_multiple_files=True)

# bot = Retriever()

def get_docs(streamlit_uploaded_files):
    documents = []
    for uploaded_file in streamlit_uploaded_files:
    #use tempfile because CSVLoader only accepts a file_path
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        file_type = uploaded_file.name.split(".")[1]
        if file_type == 'csv':
            loader = CSVLoader(tmp_file_path,csv_args = {"delimiter": ','})
        elif file_type == 'txt':
            loader = TextLoader(tmp_file_path,encoding='utf8')
        elif file_type == 'pdf':
            loader = PyPDFLoader(tmp_file_path)
        documents.extend(loader.load())

    return documents

# def conversational_chat(query):
        
#     result = chain({"question": query, "chat_history": st.session_state['history']})
#     # st.session_state['history'].append((query, result["answer"]))
    
#     return result["answer"]

def set_session():

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hi ! Please upload files and ask query from the files uploaded"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey !"]
        
    #container for the chat history
    response_container = st.container()
    #container for the user's text input
    container = st.container()

    return st.session_state, response_container, container


def main():

    uploaded_files = st.sidebar.file_uploader("upload", accept_multiple_files=True)

    bot = Retriever()

    bot.documents = get_docs(uploaded_files)

    if bot.documents:
        
        texts = bot.text_splitter.split_documents(bot.documents)

        vectorstore = Chroma.from_documents(texts, bot.embedding_model)

        docsearch = vectorstore.as_retriever(search_type="similarity",search_kwargs={"k": 2})

        bot.retriever_chain  = ConversationalRetrievalChain.from_llm(llm = bot.llm_model,retriever=docsearch)


    session_state, response_container, container = set_session()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            
            user_input = st.text_input("Query:", placeholder="Ask a query from the files uploaded", key='input')
            submit_button = st.form_submit_button(label='Send')
            
        if submit_button and user_input:

            for item in docsearch.get_relevant_documents(user_input):
                print(item)

            response = bot.retriever_chain({"question": user_input, "chat_history": st.session_state['history']})
            

            output = response["answer"]
            
            session_state['past'].append(user_input)
            session_state['generated'].append(output)


    if session_state['generated']:
        with response_container:
            for i in range(len(session_state['generated'])):
                message(session_state["past"][i], is_user=True, key=str(i) + '_user')
                message(session_state["generated"][i], key=str(i))

if __name__ == "__main__":
    main()
