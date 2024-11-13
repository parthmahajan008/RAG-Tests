from rag_tests.llm.prompt_templates import SelfQueryPromptTemplate
from langchain_openai import ChatOpenAI 
from rag_tests.config import settings
from rag_tests.llm.chain import GeneralQueryChain
from rag_tests.core import get_logger

logger = get_logger(__name__)
class SelfQuery:
    @staticmethod
    def generate_response(query:str)->str|None:
        prompt = SelfQueryPromptTemplate().create_template()
        model = ChatOpenAI(model=settings.OPENAI_MODEL_ID, temperature=0)
        chain = GeneralQueryChain().get_chain(llm=model, template=prompt, output_key="metadata_filter_value")

        response = chain.invoke({"question": query})
        result = response.get("metadata_filter_value", "none")
        if isinstance(result, str) and result.lower() == "none":
            logger.info("result is none")
            return None

        return result