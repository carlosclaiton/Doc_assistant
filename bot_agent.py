# LOADING
import os
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


class agent:
    
    def __init__(self,):
        
        # self.model="gpt-3.5-turbo"
        self.model = "gpt-4"
        
         
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
        
    def create_agent(self,):
        """"
        Define agent with memory from the chat
        """
        
        index = self.indexer()
        # Define tools
        tools = [
            Tool(
            name = "LlamaIndex",
                # func=lambda q: str(index.as_query_engine().query(q)),
                func=lambda q: str(index.as_query_engine().query(q)),
                description="useful for when you want to answer questions about the author. The input to this tool should be a complete english sentence.",
                return_direct=True
            ),
        ]
        #Initialize conversational memory
        conversational_memory = ConversationBufferWindowMemory( memory_key='chat_history', k=5, return_messages=True )
        # Initialize agent with conversational memory
        agent_executor = initialize_agent(tools, llm=ChatOpenAI(temperature=0.7, model_name=self.model), agent="conversational-react-description", memory=conversational_memory)
    
        return(agent_executor)