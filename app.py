from flask import Flask, render_template, request, jsonify
from embedding import Embedding

import os
import json
import openai

from langchain.chat_models import ChatOpenAI

## Using langchain agent
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

with open("CONFIG_LIST.json", "r") as file:
    config = json.load(file)

os.environ['OPENAI_API_KEY'] = config["openai_api_key"]
openai.api_key = os.getenv("OPENAI_API_KEY")
model = config["model"] 


app = Flask(__name__)

emb = Embedding()
index = emb.indexer()

tools = [
Tool(
    name = "LlamaIndex",
    func=lambda q: str(index.as_query_engine().query(q)),
    description="useful for when you want to answer questions about the author. The input to this tool should be a complete english sentence.",
    return_direct=True
    ),
]
#Initialize conversational memory
conversational_memory = ConversationBufferWindowMemory( memory_key='chat_history', k=5, return_messages=True )
# Initialize agent with conversational memory
agent_executor = initialize_agent(tools, llm=ChatOpenAI(temperature=0.7, model_name=model), agent="conversational-react-description", memory=conversational_memory)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get')
def ask():
    userText = request.args.get('msg')
    bot_response = bot(userText)
    return bot_response


def bot(prompt):
    
    # if I use my index
    while True:
        # prompt = input("type prompt")
        if prompt == 'thanks':
            return(f' ----------chat is closed -------')
            
        else:
            response = agent_executor.run(input=prompt)
            
            return(response)
    

if __name__ == '__main__':
    app.run(debug=True)
    
