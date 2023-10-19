# LOADING
import os
import json
from llama_index import SimpleDirectoryReader
from llama_index import LLMPredictor, PromptHelper, ServiceContext
from llama_index import GPTVectorStoreIndex
from llama_index import load_index_from_storage
from llama_index import StorageContext


from langchain.chat_models import ChatOpenAI

## Using langchain agent
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.chains.conversation.memory import ConversationBufferWindowMemory


class Embedding:
    
    def __init__(self,):
        
        # self.model="gpt-3.5-turbo"
        # self.model = "gpt-4"
        with open("CONFIG_LIST.json", "r") as file:
            config = json.load(file)
        self.model  = config["model"]
        
         
    def read_docs(self,):
        documents = SimpleDirectoryReader('MyBook').load_data() # reads whole drectory.
        return(documents)


    def indexer(self,):
        
        if os.path.exists('storage'):
            print('exist')
            
            # rebuild storage context
            storage_context = StorageContext.from_defaults(persist_dir="storage")
            # load index
            index = load_index_from_storage(storage_context)

        
        else:
            print('not exist, creating index ... ')

            documents = self.read_docs()
            
            llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name=self.model))
            # llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="text-davinci-003"))

            max_input_size = 4096
            num_output = 256
            max_chunk_overlap = 0.1
            chunk_size_limit = 600
            prompt_helper = PromptHelper(max_input_size, num_output,max_chunk_overlap,chunk_size_limit=chunk_size_limit)
            service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

            # index content in the folder documents
            index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context) 
            # Save your index to a directory called storage
            index.storage_context.persist()
        
        return(index)
        