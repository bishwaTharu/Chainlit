import os
from dotenv import load_dotenv, find_dotenv
from langchain import HuggingFaceHub
from huggingface_hub import login
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
import chainlit as cl

load_dotenv(find_dotenv())
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
login(token=api_token)

MODEL_ID = "google/flan-t5-xxl"
template = """You are a helpful chatbot answer the question {question} accurately and please provide answer politely.
{chat_history}
Human: {question}
Chatbot:"""

handler = StreamingStdOutCallbackHandler()
llm = HuggingFaceHub(
    repo_id=MODEL_ID,
    model_kwargs={"temperature": 1, "max_length": 1024},
    callbacks=[handler],
)
memory = ConversationBufferMemory(memory_key="chat_history")


@cl.langchain_factory(use_async=False)
def factory():
    prompt = PromptTemplate(
        input_variables=["question","chat_history"], template=template
    )
    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True, memory=memory)

    return llm_chain
