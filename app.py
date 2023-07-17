import os
import chainlit as cl
import faiss
from dotenv import load_dotenv, find_dotenv
from langchain import HuggingFaceHub
from huggingface_hub import login
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.experimental import AutoGPT
from langchain.agents import Tool
from langchain.docstore import InMemoryDocstore
from langchain.tools import DuckDuckGoSearchRun
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.vectorstores import FAISS


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
        description="useful for when you need to answer questions about current events. You should ask targeted questions",
    ),
    WriteFileTool(),
    ReadFileTool(),
]

embeddings_model = HuggingFaceHubEmbeddings()
embedding_size = 768
index = faiss.IndexFlatL2(embedding_size)


@cl.langchain_factory(use_async=False)
def agent():
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
    callbacks = [cl.LangchainCallbackHandler()]
    agent = AutoGPT.from_llm_and_tools(
        ai_name="Jarvis",
        ai_role="Assistant",
        tools=tools,
        llm=llm,
        memory=vectorstore.as_retriever(),

    )
    # Set verbose to be true
    agent.chain.verbose = True
    agent.chain.callbacks = callbacks
    return agent


@cl.langchain_run
async def run(agent, input):
    # Since the agent is sync, we need to make it async
    res = await cl.make_async(agent.run)([input])
    await cl.Message(content=res).send()
