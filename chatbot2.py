import os
import openai
from dotenv import load_dotenv
import requests
import textract
import tiktoken
from llama_index import download_loader, GPTSimpleVectorIndex, SimpleDirectoryReader
from pathlib import Path
from llama_index import GPTListIndex, LLMPredictor
from langchain import OpenAI
from llama_index.composability import ComposableGraph
from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from llama_index.langchain_helpers.agents import LlamaToolkit, create_llama_chat_agent, IndexToolConfig, GraphToolConfig
from llama_index.indices.query.query_transform.base import DecomposeQueryTransform


load_dotenv()

os.environ["http_proxy"] = "http://10.10.1.3:10000"
os.environ["https_proxy"] = "http://10.10.1.3:10000"

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_API_KEY'] = openai.api_key


'''
# data preprocessing => index docs
doc_ids = [361, 466, 541, 684, 715]
PDFReader = download_loader("PDFReader")
loader = PDFReader()

doc_set = {}
all_docs = []
for doc_id in doc_ids:
    id_docs = loader.load_data(file=Path(f'./data/radiolarian/{doc_id}.pdf'))
    # insert year metadata into each document
    for d in id_docs:
        d.extra_info = {"Doc ID": doc_id}
    doc_set[doc_id] = id_docs
    all_docs.extend(id_docs)

# initialize simple vector indices + global vector index
print("# indexing...")
index_set = {}
for doc_id in doc_ids:
    cur_index = GPTSimpleVectorIndex(doc_set[doc_id], chunk_size_limit=512)
    index_set[doc_id] = cur_index
    cur_index.save_to_disk(f'./data/indexed/radiolarian_index_{doc_id}.json')
print("# index finished")
'''


# Load indices from disk
doc_ids = [361, 466, 541, 684, 715]
index_set = {}
for doc_id in doc_ids:
    cur_index = GPTSimpleVectorIndex.load_from_disk(f'./data/indexed/radiolarian_index_{doc_id}.json')
    index_set[doc_id] = cur_index

# set summary text for each doc
for doc_id in doc_ids:
    index_set[doc_id].set_text(f"Radialarian Paper ID {doc_id}")

# define an LLMPredictor set number of output tokens
llm_predictor = LLMPredictor(llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=512))

# define a list index over the vector indices
# allows us to synthesize information across each index
list_index = GPTListIndex([index_set[doc_id] for doc_id in doc_ids], llm_predictor=llm_predictor)

# graph = ComposableGraph.build_from_index(list_index)

# [optional] save to disk
# graph.save_to_disk('./data/indexed/radiolarian_graph.json')
# [optional] load from disk, so you don't need to build graph from scratch
graph = ComposableGraph.load_from_disk('./data/indexed/radiolarian_graph.json', llm_predictor=llm_predictor)


# define a decompose transform
decompose_transform = DecomposeQueryTransform(
    llm_predictor, verbose=True
)

# define query configs for graph 
query_configs = [
    {
        "index_struct_type": "simple_dict",
        "query_mode": "default",
        "query_kwargs": {
            "similarity_top_k": 1,
            # "include_summary": True
        },
        "query_transform": decompose_transform
    },
    {
        "index_struct_type": "list",
        "query_mode": "default",
        "query_kwargs": {
            "response_mode": "tree_summarize",
            "verbose": True
        }
    },
]

# graph config
graph_config = GraphToolConfig(
    graph=graph,
    name=f"Graph Index",
    description="useful for when you want to answer queries that require analyzing multiple Radiolarian documents.",
    query_configs=query_configs,
    tool_kwargs={"return_direct": True}
)


# define toolkit
index_configs = []
for y in [361, 466, 541, 684, 715]:
    tool_config = IndexToolConfig(
        index=index_set[y], 
        name=f"Vector Index {y}",
        description=f"useful for when you want to answer queries about the {y}.pdf radiolarian document",
        index_query_kwargs={"similarity_top_k": 3},
        tool_kwargs={"return_direct": True}
    )
    index_configs.append(tool_config)

toolkit = LlamaToolkit(
    index_configs=index_configs,
    graph_configs=[graph_config]
)

memory = ConversationBufferMemory(memory_key="chat_history")
llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
agent_chain = create_llama_chat_agent(
    toolkit,
    llm,
    memory=memory,
    verbose=True
)


agent_chain.run(input="hi, i am bob")

query_str = 'Extract key pieces of information from all the indexed radiolarian files across [361, 466, 541, 684, 715].\nIf a particular piece of information is not present, output \"Not specified\".\nWhen you extract a key piece of information, include the closest page number.\nUse the following format:\n0. What is the title\n1. What is the location or address related to boulder, samples and radiolarians\n2. Which sentence is in the form of "XXX (boulder, samples, radiolarians) was/were collected in/from\n\nDocument: \"\"\"{document}\"\"\"\n\n0. Who is the title: paper title (Page 1)\n1.'
print(f'Input: {query_str}')
response = agent_chain.run(input=query_str)
print(f'Agent: {response}')


# while True:
#     text_input = input("User: ")
#     response = agent_chain.run(input=text_input)
#     print(f'Agent: {response}')
