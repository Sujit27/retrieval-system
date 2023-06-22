import os
import config

from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


class Retriever:
    '''
    '''
    def __init__(self,embedding_model="all-MiniLM-L6-v2",llm_model='open-ai'):
        
        self.text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200,separator = " ")
        self.documents = None

        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)

        if llm_model == 'open-ai':
            self.llm_model = ChatOpenAI(temperature=0.0,model_name='gpt-3.5-turbo')
        else:
            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            # Make sure the model path is correct for your system!
            self.llm_model = LlamaCpp(model_path="../models/vicuna-7b-1.1.ggmlv3.q4_0.bin", callback_manager=callback_manager, verbose=True, n_ctx=1024)

        self.retriever_chain = None

    