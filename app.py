from flask import Flask, render_template, request, jsonify
from bot_agent import agent  # Replace with your actual chatbot module

import os
import json
import openai

with open("key.json", "r") as file:
    config = json.load(file)

os.environ['OPENAI_API_KEY'] = config["openai_api_key"]
openai.api_key = os.getenv("OPENAI_API_KEY") 

bot = agent()
agent_executor = bot.create_agent()

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get')
def ask():
    userText = request.args.get('msg')
    bot_response = agent_executor.run(input=userText) 
    return bot_response

if __name__ == '__main__':
    app.run(debug=True)
    
