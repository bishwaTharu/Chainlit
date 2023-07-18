import os
import chainlit as cl
from dotenv import load_dotenv, find_dotenv
from langchain import HuggingFaceHub
from huggingface_hub import login
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.agents import Tool
from langchain.tools import DuckDuckGoSearchRun
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent
from langchain.agents import AgentType


# authentication
load_dotenv(find_dotenv())
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
login(token=api_token)

# model id
MODEL_ID = "google/flan-t5-xxl"

handler = StreamingStdOutCallbackHandler()
llm = HuggingFaceHub(
    repo_id=MODEL_ID,
    model_kwargs={"temperature": 1, "max_length": 1024},
    callbacks=[handler],
)


search = DuckDuckGoSearchRun()


tools = [
    Tool(
        name="DuckDuckGo Search",
        func=search.run,
        description="useful for when you need to answer questions about current events or the current state of the world",
    )
]


@cl.langchain_factory(use_async=False)
def agent():
    memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
    agent_chain = initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        max_execution_time=1,
        early_stopping_method="generate",
        handle_parsing_errors="Check your output and make sure it conforms!",

    )

    # Set verbose to be true
    return agent_chain


@cl.langchain_run
async def run(agent, input_str):
    # Since the agent is sync, we need to make it async
    res = await cl.make_async(agent.run)(input_str)
    await cl.Message(content=res).send()
