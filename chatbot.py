import os
import openai
from dotenv import load_dotenv
import requests
from llama_index import download_loader, GPTSimpleVectorIndex
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
years = [2022, 2021, 2020, 2019]
UnstructuredReader = download_loader("UnstructuredReader", refresh_cache=True)
loader = UnstructuredReader()

doc_set = {}
all_docs = []
for year in years:
    year_docs = loader.load_data(file=Path(f'./data/UBER/UBER_{year}.html'), split_documents=False)
    # insert year metadata into each year
    for d in year_docs:
        d.extra_info = {"year": year}
    doc_set[year] = year_docs
    all_docs.extend(year_docs)

# initialize simple vector indices + global vector index
print("# indexing...")
index_set = {}
for year in years:
    cur_index = GPTSimpleVectorIndex(doc_set[year], chunk_size_limit=512)
    index_set[year] = cur_index
    cur_index.save_to_disk(f'./data/indexed/index_{year}.json')
print("# index finished")
'''


# Load indices from disk
years = [2022, 2021, 2020, 2019]
index_set = {}
for year in years:
    cur_index = GPTSimpleVectorIndex.load_from_disk(f'./data/indexed/index_{year}.json')
    index_set[year] = cur_index

# set summary text for each doc
for year in years:
    index_set[year].set_text(f"UBER 10-k Filing for {year} fiscal year")

# define an LLMPredictor set number of output tokens
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, max_tokens=512))

# define a list index over the vector indices
# allows us to synthesize information across each index
list_index = GPTListIndex([index_set[y] for y in years], llm_predictor=llm_predictor)

# graph = ComposableGraph.build_from_index(list_index)

# [optional] save to disk
# graph.save_to_disk('./data/indexed/10k_graph.json')
# [optional] load from disk, so you don't need to build graph from scratch
graph = ComposableGraph.load_from_disk('./data/indexed/10k_graph.json', llm_predictor=llm_predictor)


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
    description="useful for when you want to answer queries that require analyzing multiple SEC 10-K documents for Uber.",
    query_configs=query_configs,
    tool_kwargs={"return_direct": True}
)


# define toolkit
index_configs = []
for y in range(2019, 2023):
    tool_config = IndexToolConfig(
        index=index_set[y], 
        name=f"Vector Index {y}",
        description=f"useful for when you want to answer queries about the {y} SEC 10-K for Uber",
        index_query_kwargs={"similarity_top_k": 3},
        tool_kwargs={"return_direct": True}
    )
    index_configs.append(tool_config)

toolkit = LlamaToolkit(
    index_configs=index_configs,
    graph_configs=[graph_config]
)

memory = ConversationBufferMemory(memory_key="chat_history")
llm=OpenAI(temperature=0)
agent_chain = create_llama_chat_agent(
    toolkit,
    llm,
    memory=memory,
    verbose=True
)


agent_chain.run(input="hi, i am bob")
agent_chain.run(input="What were some of the biggest risk factors in 2020 for Uber?")
cross_query_str = (
    "Compare/contrast the risk factors described in the Uber 10-K across years. Give answer in bullet points."
)
agent_chain.run(input=cross_query_str)


# while True:
#     text_input = input("User: ")
#     response = agent_chain.run(input=text_input)
#     print(f'Agent: {response}')
