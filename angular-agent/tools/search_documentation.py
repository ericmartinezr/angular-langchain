import bs4
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import tool
from dotenv import load_dotenv
from utils.logger import logger

load_dotenv()

# https://angular.dev/llms.txt
# https://angular.dev/context/llm-files/llms-full.txt
DOCUMENTATION_URL = [
    "https://angular.dev/context/llm-files/llms-full.txt",
    "https://angular.dev/style-guide"]


async def search_documentation():
    """
    Searches the official Angular documentation for information about components, directives,
    services, APIs, template syntax, and code examples.
    Use this tool when you need to understand Angular concepts, find usage patterns,
    or get code snippets to generate Angular applications or components.

    Args:
    """
    logger.debug("search_documentation init")

    retriever = None
    try:
        loader = WebBaseLoader(web_paths=DOCUMENTATION_URL, verify_ssl=True, bs_kwargs={
            "parse_only": bs4.SoupStrainer("docs-viewer")
        })

        documentation = []
        async for doc in loader.alazy_load():
            documentation.append(doc)

        assert len(documentation) == 1

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = InMemoryVectorStore.from_documents(
            documentation, embeddings)

        retriever = create_retriever_tool(
            vector_store.as_retriever(),
            name="search_documentation",
            description=(
                """
                Searches the official Angular documentation for information about components, directives,
                services, APIs, template syntax, and code examples.
                Use this tool when you need to understand Angular concepts, find usage patterns,
                or get code snippets to generate Angular applications or components.
                """.strip()
            )
        )
    except Exception as e:
        logger.error("Error when searching documentation")
        logger.error(e)

    logger.debug("search_documentation end")

    return retriever
