import os
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


# user_api_key = st.sidebar.text_input(
#     label="#### Your OpenAI API key 👇",
#     placeholder="Paste your openAI API key, sk-",
#     type="password")

os.environ["OPENAI_API_KEY"] = ''

uploaded_files = st.sidebar.file_uploader("upload", accept_multiple_files=True)

documents = []
for uploaded_file in uploaded_files:
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

if uploaded_files:
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200,separator = " ")
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(texts, embeddings)

    chain = ConversationalRetrievalChain.from_llm(llm = ChatOpenAI(temperature=0.0,model_name='gpt-3.5-turbo'),retriever=vectorstore.as_retriever())

def conversational_chat(query):
        
    result = chain({"question": query, "chat_history": st.session_state['history']})
    # st.session_state['history'].append((query, result["answer"]))
    
    return result["answer"]

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

with container:
    with st.form(key='my_form', clear_on_submit=True):
        
        user_input = st.text_input("Query:", placeholder="Ask a query from the files uploaded", key='input')
        submit_button = st.form_submit_button(label='Send')
        
    if submit_button and user_input:
        try:
            output = conversational_chat(user_input)
            
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
        except:
            pass

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
