import os
import json
import openai
import chainlit as cl

from embedding import Embedding

with open("CONFIG_LIST.json", "r") as file:
    config = json.load(file)

os.environ['OPENAI_API_KEY'] = config["openai_api_key"]
openai.api_key = os.getenv("OPENAI_API_KEY")
model = config["model"] 


## Function to load the index from storage or create a new one
@cl.cache  ## Allow to cache the function
def load_context():
    emb = Embedding()
    index = emb.indexer()
    return(index)

@cl.on_chat_start
async def start():
    index = load_context()

    query_engine = index.as_query_engine(
        streaming=True,
        similarity_top_k=2,
    )

    cl.user_session.set("query_engine", query_engine)

    await cl.Message(author="Assistant", content="Hello ! How may I help you ? ").send()


@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine")

    msg = cl.Message(content="", author="Assistant")

    res = query_engine.query(message.content)

    for text in res.response_gen:
        token = text
        await msg.stream_token(token)

    await msg.send()