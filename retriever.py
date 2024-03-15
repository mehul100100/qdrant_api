from llama_index.core import QueryBundle
from qdrant_client import QdrantClient
from llama_index.core.retrievers import BaseRetriever
from typing import Any, List
from llama_index.core.schema import NodeWithScore
from typing import Optional
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.vector_stores.qdrant import QdrantVectorStore

client = QdrantClient(url="https://c3914f95-6469-48b0-88bf-8e5eebf22c4e.us-east4-0.gcp.cloud.qdrant.io", api_key="7G-Kg43RzP8tQuo3oIt-YAYmI-y0t1GEwdlBbscYsqaCjsYvK8yUBA")
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")


class VectorDBRetriever(BaseRetriever):
    """Retriever over a postgres vector store."""

    def __init__(
        self,
        vector_store: Any,
        embed_model: Any,
        query_mode: str = "default",
        similarity_top_k: int = 2,
    ) -> None:
        """Init params."""
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        query_embedding = embed_model.get_query_embedding(
            query_bundle.query_str
        )
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )
        query_result = self._vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        return nodes_with_scores