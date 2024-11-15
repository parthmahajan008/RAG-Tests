from langchain_openai import ChatOpenAI

from rag_tests.llm.chain import GeneralQueryChain
from rag_tests.llm.prompt_templates import RerankingTemplate
from rag_tests.config import settings
from rag_tests.core import get_logger
logger = get_logger(__name__)

class Reranker:
    @staticmethod
    def generate_response(
        query: str, passages: list[str], keep_top_k: int
    ) -> list[str]:
        reranking_template = RerankingTemplate()
        logger.info(f"Creating Reranking Prompt Template with keep_top_k: {keep_top_k}")
        prompt_template = reranking_template.create_template(keep_top_k=keep_top_k)

        model = ChatOpenAI(model=settings.OPENAI_MODEL_ID, temperature=0)
        chain = GeneralQueryChain().get_chain(
            llm=model, output_key="rerank", template=prompt_template
        )

        stripped_passages = [
            stripped_item for item in passages if (stripped_item := item.strip())
        ]
        passages = reranking_template.separator.join(stripped_passages)
        response = chain.invoke({"question": query, "passages": passages})

        result = response["rerank"]
        reranked_passages = result.strip().split(reranking_template.separator)
        stripped_passages = [
            stripped_item
            for item in reranked_passages
            if (stripped_item := item.strip())
        ]
        return stripped_passages