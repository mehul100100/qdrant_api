from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
import os
from retriever import VectorDBRetriever

from dotenv import load_dotenv

load_dotenv()
loader = PyMuPDFReader()

client = QdrantClient(url="https://c3914f95-6469-48b0-88bf-8e5eebf22c4e.us-east4-0.gcp.cloud.qdrant.io", api_key="7G-Kg43RzP8tQuo3oIt-YAYmI-y0t1GEwdlBbscYsqaCjsYvK8yUBA")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
# documents = SimpleDirectoryReader("./data/").load_data()
vector_store = QdrantVectorStore(client=client, collection_name="invoices")

# documents = loader.load(file_path="./data/Invoice # 2.pdf")

# # Use a Text Splitter to Split Documents

# text_parser = SentenceSplitter(
#     chunk_size=1024,
#     # separator=" ",
# )

# text_chunks = []
# doc_idxs = []

# for doc_idx, doc in enumerate(documents):
#     cur_text_chunks = text_parser.split_text(doc.text)
#     text_chunks.extend(cur_text_chunks)
#     doc_idxs.extend([doc_idx] * len(cur_text_chunks))



# # Manually Construct Nodes from Text Chunks

# nodes = []
# for idx, text_chunk in enumerate(text_chunks):
#     node = TextNode(
#         text=text_chunk,
#     )
#     src_doc = documents[doc_idxs[idx]]
#     node.metadata = src_doc.metadata
#     nodes.append(node)





# # Generate Embeddings for each Node
# for node in nodes:
#     node_embedding = embed_model.get_text_embedding(
#         node.get_content(metadata_mode="all")
#     )
#     node.embedding = node_embedding

# vector_store.add(nodes)

# query_str = "Can you tell me about the key concepts for safety finetuning"


# Generate Embedding for the Query
# The query_str is used as input and is embedded using the HuggingFaceEmbedding model
# query_embedding = embed_model.get_query_embedding(query_str)

# from llama_index.core.vector_stores import VectorStoreQuery
# query_mode = "default"

# vector_store_query = VectorStoreQuery(
#     query_embedding=query_embedding, similarity_top_k=2, mode=query_mode
# )

# query_result = vector_store.query(vector_store_query)

# print(query_result.nodes[0].get_content())

# from llama_index.core.schema import NodeWithScore
# from typing import Optional

# nodes_with_scores = []
# for index, node in enumerate(query_result.nodes):
#     score: Optional[float] = None
#     if query_result.similarities is not None:
#         score = query_result.similarities[index]
#     nodes_with_scores.append(NodeWithScore(node=node, score=score))

# print(nodes_with_scores)
######################################################

from llama_index.core.query_engine import RetrieverQueryEngine
retriever = VectorDBRetriever(
    vector_store, embed_model, query_mode="default", similarity_top_k=2
)





from llama_index.llms.openai import OpenAI

llm = OpenAI(model_name="gpt-3.5-turbo")

query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)

query_str = "What is the due date of Meta Corps Invoice?"

response = query_engine.query(query_str)

print(str(response))
# print(response.source_nodes[0].get_content())

##############################################################3