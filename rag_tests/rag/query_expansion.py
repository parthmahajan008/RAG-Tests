from langchain_openai import ChatOpenAI

from rag_tests.llm.chain import GeneralQueryChain
from rag_tests.llm.prompt_templates import QueryExpansionPromptTemplate
from rag_tests.config import settings
from rag_tests.core import get_logger

logger = get_logger(__name__)

class QueryExpansion:
    @staticmethod
    def generate_response(query: str, to_expand_to_n: int) -> list[str]:
        query_expansion_template = QueryExpansionPromptTemplate()
        logger.info(f"Creating Query Expansion Prompt Template with to_expand_to_n: {to_expand_to_n}")
        prompt_template = query_expansion_template.create_template(to_expand_to_n)
        model = ChatOpenAI(model=settings.OPENAI_MODEL_ID, temperature=0)

        chain = GeneralQueryChain().get_chain(
            llm=model, output_key="expanded_queries", template=prompt_template
        )

        response = chain.invoke({"query": query})
        result = response["expanded_queries"]

        queries = result.strip().split(query_expansion_template.separator)
        stripped_queries = [
            stripped_item for item in queries if (stripped_item := item.strip())
        ]

        return stripped_queries