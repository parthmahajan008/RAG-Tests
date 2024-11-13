from abc import ABC, abstractmethod
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from rag_tests.core import get_logger

logger = get_logger(__name__)
class BasePromptTemplate(ABC, BaseModel):
    @abstractmethod
    def create_template(self) -> PromptTemplate:
        pass

class QueryExpansionPromptTemplate(BasePromptTemplate):
    prompt:str = """You are an AI language model assistant. You are being used a topic expanding service, in a 
    Rag system. The user query would describe a topic, user wants to generate a linkedin post about. 
    based on the user query, you need to generate {to_expand_to_n} topics as "english statements" aka queries to make a search in a vector database.
    These queries should be similar to the user query.
    
    Your task is to generate {to_expand_to_n}
    different versions of the given user's topic to retrieve relevant documents from a vector
    database. By generating multiple perspectives on the user topics, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search in a vector database.
    The vector database contains metadata of posts, along with their embeddings. The embeddings are based on the posts' content.
    Provide these queries to search in the vector database, seperated by '{separator}'.

    User query: {query}"""

    @property
    def separator(self) -> str:
        return "#next-question#"
    
    def create_template(self, to_expand_to_n:int) -> PromptTemplate:
        logger.info(f"Creating Query Expansion Prompt Template with to_expand_to_n: {to_expand_to_n} and separator: {self.separator}")
        template =  PromptTemplate(
            template=self.prompt,
            input_variables=["query"],
            partial_variables={
                "separator": self.separator,
                "to_expand_to_n": to_expand_to_n,
            },
        )
        logger.info(f"Query Expansion Prompt Template created: \033[93m{template}\033[0m")
        return template


# TODO : ToImprove : Add Personlisation metadata- like user_name, experience details, learnings, events, perhaps location etc
class SelfQueryPromptTemplate(BasePromptTemplate):
    prompt:str ="""You are a AI language model assistant. Your task is to extract information from a user question. The required information
    that needs to be extracted is the author_id. Your response should consist of only the extracted author_id (eg : https://linkedin.com/in/varunAgarwal2)\
    If you cannot find the author_id, return "None"
    User question: {question}"""

    def create_template(self) -> PromptTemplate:
        template = PromptTemplate(
            template=self.prompt,
            input_variables=["question"],
            output_variables=["author_id"],
        )
        logger.info(f"Self Query Prompt Template created: \033[93m{template}\033[0m")
        return template

class QueryRefinementPromptTemplate(BasePromptTemplate):
    prompt:str = """You are a AI language model assistant. Your task is to refine a user query. The user query is: {query}"""

    def create_template(self) -> PromptTemplate:
        template = PromptTemplate(
            template=self.prompt,
            input_variables=["query"],
        )
        logger.info(f"Query Refinement Prompt Template created: \033[93m{template}\033[0m")
        return template
    
class ExtractToneFromHitsPromptTemplate(BasePromptTemplate):
    prompt:str = """You are a content assistant specialized in analyzing text to extract tone, style, and formatting guidelines. 
    Your goal is to help identify consistent patterns and rules in a user’s writing style by examining their past content.
      Based on these insights, generate specific rules for style, tone, vocabulary, sentence structure, and formatting to ensure 
      future content aligns with the user's established voice.
      Analyze the following posts and derive a set of rules for generating content that aligns with my writing style and tone. Include guidelines for:  
        1.Style and Tone (formal, conversational, persuasive).
        2.Vocabulary Preferences (preferred words/phrases, jargon to avoid).
        3.Sentence Structure (active vs. passive voice, sentence length variation).
        4.Platform-Specific Formatting (LinkedIn, blog posts, emails).
        5.Inclusivity & Accessibility (gender-neutral language, readability).
        6.Semantic Rules (context-aware tone adjustments, avoid repetition).
        7.Consistency (punctuation, capitalization, formatting).
        Here are USER's past posts:
        <begin_posts> {posts} <end_posts>
        Provide the extracted rules in a structured format with examples where applicable. RETURN DATA IN JSON FORMAT ONLY.
      """

    def create_template(self) -> PromptTemplate:
        template = PromptTemplate(
            template=self.prompt,
            input_variables=["posts"],
            output_parser=JsonOutputParser(),
        )
        logger.info(f"Extract Tone From Hits Prompt Template created: \033[93m{template}\033[0m")
        return template
    
class GetToneUnderstandingPromptTemplate(BasePromptTemplate):
    prompt:str = """You are a helpful AI assistant, that understands social media, words, and grammar very well. We have
    retrieved relevant documents from a vector database, and we want you to understand the tone, patterns and voice of the user.
    The user is : {author_id}.
    Based on the recent posts by the user, gather insights out of these posts:- \n <begin_posts> {posts} <end_posts>. 
    Identify the tone, patterns and voice in these posts and return them in a proper format,
    JSON(tone : str, patterns : str, voice : str).
    Patterns should include things like 
    1. whether the author uses emojis, 
    2. mentions how many hashtags, 
    3. understanding of message ending, middle and ending, 
    4. understand the flow of words and paragraphs, 
    5. when does the author change the paragraph 
    6. how does the author use the hashtags, emojis, and other symbols
    7. What length of paragraphs does the author use
    8. What is the use of line breaks, new lines by the author
    9. What is the overal length of the posts etc 

    Be highly qualitative as possible in pattern understanding

    All three are descriptive attributes of the posts, keep them clear and high quality
    JSON(
    "tone": "Informative, engaging, and motivational, with a professional tone that reflects expertise and enthusiasm for AI and startups. The tone is authoritative yet conversational, with clear calls to action (e.g., 'Apply now,' 'Watch the show').",
  
  "patterns": "The posts are structured with a clear beginning that introduces the topic, a middle with key points or event details, and an ending that often encourages interaction or gives a call to action. Posts are long and segmented with line breaks for readability, using short to medium-length paragraphs that allow for easy scanning of information. Paragraphs change frequently, often to introduce new ideas or points. Hashtags are sparingly used, typically no more than 2-3 per post, and are applied to highlight specific keywords such as #AI and #LinkedInTopStartups. No emojis are used, keeping the tone formal. URLs are included towards the end for further engagement, and posts often reference well-known personalities or organizations to build authority. The posts conclude with engagement prompts like 'Let me know your thoughts.' Each post is lengthy but broken up with deliberate line breaks, which adds visual rhythm and reinforces key points effectively.",

    "voice": "Authoritative and supportive, with a mentorship-like approach. The voice is that of a knowledgeable industry leader who aims to educate and inspire the reader. It balances insight and practical advice, often emphasizing the importance of testing ideas, deep understanding, and strategic partnerships. There’s also a slight promotional tone when referencing Change Engine’s offerings and content, which is blended naturally within the educational style."
    )

    """
    class ToneAnalysis(BaseModel):
        tone: str
        patterns: str
        voice: str

    def create_template(self) -> PromptTemplate:
        template = PromptTemplate(
            template=self.prompt,
            input_variables=["author_id", "posts"],
            output_parser=JsonOutputParser(pydantic_object=self.ToneAnalysis),
        )
        logger.info(f"Get Tone Understanding Prompt Template created: \033[93m{template}\033[0m")
        return template
    
class RerankingTemplate(BasePromptTemplate):
    prompt: str = """
    You are an AI language model assistant. Your task is to rerank passages related to a 
    query based on their relevance. 
    The most relevant passages should be put at the beginning. 
    You should only pick at max {keep_top_k} passages.
    The provided and reranked documents are separated by '{separator}'.
    
    The following are passages related to this query: {question}.
    
    Passages: 
    {passages}
    """

    def create_template(self, keep_top_k: int) -> PromptTemplate:
        template = PromptTemplate(
            template=self.prompt,
            input_variables=["question", "passages"],
            partial_variables={"keep_top_k": keep_top_k, "separator": self.separator},
        )
        logger.info(f"Reranking Prompt Template created: \033[93m{template}\033[0m")
        return template

    @property
    def separator(self) -> str:
        return "\n#next-document#\n"
