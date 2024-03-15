from flask import Flask, request, jsonify
from flask_cors import CORS
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI
from retriever import VectorDBRetriever
import requests
import os
from dotenv import load_dotenv

load_dotenv()
loader = PyMuPDFReader()

app = Flask(__name__)
CORS(app, resources={r'/*': {'origins': '*'}})

client = QdrantClient(url=os.environ.get("QDRANT_URL"), api_key=os.environ.get("QDRANT_API_KEY"))
# embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")




@app.route('/')
def hello_world():
    return {"Hello":"World"}


@app.route('/add_pdf', methods=['POST'])
def add_pdf():
    try:
        file = request.files['file']
        user_id = request.form.get('user_id')
        if file and file.filename.endswith('.pdf'):
            filename = os.path.join(os.getcwd(), file.filename)
            file.save(filename)
            collection_name = user_id.split('@')[0] + '_invoice_vector'
            vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
            documents = loader.load(file_path=filename)

            text_parser = SentenceSplitter(
                chunk_size=1024,
            )

            text_chunks = []
            doc_idxs = []

            for doc_idx, doc in enumerate(documents):
                cur_text_chunks = text_parser.split_text(doc.text)
                text_chunks.extend(cur_text_chunks)
                doc_idxs.extend([doc_idx] * len(cur_text_chunks))

            nodes = []
            for idx, text_chunk in enumerate(text_chunks):
                node = TextNode(
                    text=text_chunk,
                    metadata=documents[doc_idxs[idx]].metadata,
                )
                node_embedding = embed_model.get_text_embedding(node.get_content(metadata_mode="all"))
                node.embedding = node_embedding
                nodes.append(node)

            vector_store.add(nodes)
            
            print("user_id", user_id)
            os.remove(filename)

            return {"status": "success"}, 200
    except Exception as e:
        print(e)
        return {"status": "error"}, 500


@app.route('/query', methods=['POST'])
def query_api():
    try:
        query_str = request.json.get('query')
        user_id = request.json.get('user_id')
        collection_name = user_id.split('@')[0] + '_invoice_vector'
        vector_store = QdrantVectorStore(client=client, collection_name=collection_name)

        llm = OpenAI(model_name="gpt-3.5-turbo")

        retriever = VectorDBRetriever(
            vector_store, embed_model, query_mode="default", similarity_top_k=2
        )
        query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)
        res = query_engine.query(query_str)
        

        return jsonify({"status": "success", "data": str(res)}), 200
    except Exception as e:
        print(e)
        return jsonify({"status": "fail"}), 500







if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

