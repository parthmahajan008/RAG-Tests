import concurrent.futures #for running multiple queries in parallel
from rag_tests.core.logger_utils import get_logger
from  rag_tests.utils import flatten
from rag_tests.qdrant_db import QdrantDatabaseConnector
from qdrant_client import models
from sentence_transformers import SentenceTransformer
from rag_tests.config import settings
from rag_tests.rag.query_expansion import QueryExpansion
from rag_tests.rag.self_query import SelfQuery
from rag_tests.rag.reranking import Reranker


logger = get_logger(__name__)


class VectorRetriever:
    """
    class for retreiving vectors from qdrant store in a RAG system using query expansion and multitenancy search
    """

    def __init__(self, query:str, ) -> None:
        self.client = QdrantDatabaseConnector()
        self.query = query
        self._embedder = SentenceTransformer(settings.EMBEDDING_MODEL_ID)
        self._query_expander = QueryExpansion()
        self._metadata_extractor = SelfQuery()
        self._reranker = Reranker()
        logger.info(f"VectorRetriever initialized with query: \033[93m{query}\033[0m")

    def _search_single_query(
            self, generated_query:str, metadata_filter_value:str|None, k:int, collection_name:str
    ):
        assert k>3, "k must be greater than 3"
        assert k<100, "k must be less than 100"
        
        logger.info(f"Searching single query: {generated_query} and metadata filter value: {metadata_filter_value}")

        query_vector = self._embedder.encode(generated_query).tolist()
        query_filter = None
        if metadata_filter_value:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="owner_id",
                        match=models.MatchValue(value=metadata_filter_value)
                    )
                ]
            )
            logger.debug(f"Query filter created: {query_filter}")

        vectors = [
            self.client.search(
                collection_name=collection_name,
                query_filter=query_filter,
                query_vector=query_vector,
                limit=k//3
            )
        ]
        logger.info(f"Search completed. Number of vectors: \033[93m{len(vectors[0])}\033[0m")

        return flatten(vectors)

    
    def set_query(self, query:str):
        self.query = query


    def get_query(self):
        return self.query

    def set_embedder(self, embedder:SentenceTransformer):
        self._embedder = embedder


    def get_embedder(self):

        return self._embedder


    def retrieve_top_k(self, k: int, to_expand_to_n_queries: int, collection_name:str) -> list:
        logger.info(f"Retrieving top {k} results, expanding to {to_expand_to_n_queries} queries")
        logger.info("Calling QueryExpansion.generate_response")
        generated_queries = self._query_expander.generate_response(
            self.query, to_expand_to_n=to_expand_to_n_queries
        )
        logger.info(f"Generated Expanded queries: \033[93m{generated_queries}\033[0m")

        logger.info("Calling SelfQuery.generate_response to extract author_id")
        author_id = self._metadata_extractor.generate_response(self.query)
        if author_id:
            logger.info(
                f"Successfully extracted the author_id from the query: \033[93m{author_id}\033[0m"
            )
        else:
            logger.info("Couldn't extract the author_id from the query.")
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            search_tasks = [
                executor.submit(self._search_single_query, query, metadata_filter_value=author_id, k=k, collection_name=collection_name)
                for query in generated_queries
            ]

            hits = [
                task.result() for task in concurrent.futures.as_completed(search_tasks)
            ]
            hits = flatten(hits)

        logger.info(f"All documents retrieved successfully. Number of documents: {len(hits)}")
        logger.info(f"vectorRetriever.retrieve_top_k() Hits: \033[93m{hits}\033[0m")

        return hits

    def rerank(self, hits: list, keep_top_k: int) -> list[str]:
        logger.info(f"Reranking {len(hits)} hits, keeping top {keep_top_k}")
        content_list = [hit.payload["content"] for hit in hits]
        logger.info("Calling Reranker.generate_response")
        rerank_hits = self._reranker.generate_response(
            query=self.query, passages=content_list, keep_top_k=keep_top_k
        )

        logger.info(f"Documents reranked successfully. Number of documents: {len(rerank_hits)}")

        return rerank_hits