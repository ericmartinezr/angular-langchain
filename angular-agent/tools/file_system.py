import os
import bs4
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.tools import tool
from langchain.tools.retriever import create_retriever_tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from schemas.file import FileGenerated
from dotenv import load_dotenv
from utils.logger import logger

load_dotenv()

PROJECT_OUTPUT = "/home/eric/langchain-test/project_output"
DOCUMENTATION_URL = [
    "https://angular.dev/reference/configs/file-structure",
    "https://angular.dev/reference/configs/workspace-config",
    "https://angular.dev/reference/versions"
]


async def search_documentation():
    """
    Searches the official Angular documentation for comprehensive information about
    standard project structure, configuration files (e.g., angular.json, package.json),
    and file/directory naming conventions (e.g., kebab-case, module organization).
    **Call this tool once to establish a foundational understanding of Angular project setup.**
    Use the retrieved information to guide all subsequent file and directory creation.
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
                Searches the official Angular documentation for comprehensive information about
                standard project structure, configuration files (e.g., angular.json, package.json),
                and file/directory naming conventions (e.g., kebab-case, module organization).
                **Call this tool once to establish a foundational understanding of Angular project setup.**
                Use the retrieved information to guide all subsequent file and directory creation.
                """.strip()
            )
        )
    except Exception as e:
        logger.error("Error when searching documentation")
        logger.error(e)
    finally:
        logger.debug("search_documentation end")

    return retriever


@tool(parse_docstring=True)
def create_files(files: dict[str, str]) -> list[str]:
    """
    Function that creates multiple files in the predefined folder.
    Returns a list of files that could not be created.

    Args:
        files: dict[str, str] (dictionary with file names as keys and file content as values).
    """
    logger.debug("create_files init")
    logger.debug("Files to create")
    logger.debug(files)

    not_created = []
    for file_name, file_content in files.items():
        if not _create_file(file_name, file_content):
            not_created.append(file_name)

    logger.debug("Files not created")
    logger.debug(not_created)
    logger.debug("create_files end")

    return not_created


@tool(parse_docstring=True)
def create_files_with_schema(files: list[FileGenerated]) -> list[str]:
    """
    Function that creates files in the predefined folder using the FileGenerated schema.

    Args:
        files: list[FileGenerated] (List of files to be created).
    """
    logger.debug("create_files_with_schema init")
    logger.debug("Files to create")
    logger.debug(files)

    not_created = []
    for file in files:
        if not _create_file(file.path, file.content):
            not_created.append(file.path)

    logger.debug("Files not created")
    logger.debug(not_created)
    logger.debug("create_files_with_schema end")

    return not_created


@tool(parse_docstring=True)
def check_files(file_names: list[str]) -> list[str]:
    """
    Function that checks if a list of files exist in the predefined folder.
    Returns a list of file names that do not exist.

    Args:
        file_names: list[str] (list of file names to be checked).
    """
    logger.debug("check_files init")
    logger.debug("Files to check existence")
    logger.debug(file_names)

    not_exists = []
    for file_name in file_names:
        if not os.path.isfile(file_name):
            not_exists.append(file_name)

    if len(not_exists) > 0:
        logger.debug("Files not created")
        logger.debug(not_exists)
    else:
        logger.debug("All files exist")

    logger.debug("create_files_with_schema end")

    return not_exists


@tool(parse_docstring=True)
def read_file(file_name: str) -> str | None:
    """
    Function that reads a file

    Args:
        file_name: str (name of the file to be read).
    """
    logger.debug("read_file init")
    logger.debug("File to read")
    logger.debug(file_name)

    content = None
    try:
        with open(f"{PROJECT_OUTPUT}/{file_name}", mode='r', encoding='utf-8') as file:
            content = file.read()
    except Exception as e:
        logger.error(
            f"Error when reading file {file_name} in directory {PROJECT_OUTPUT}")
        logger.error(e)

    logger.debug("file_name end")

    return content


def _create_file(file_name: str, file_content: str) -> bool:
    """
    Function that creates a file in the predefined folder

    Args:
        file_name: str (name of the file to be created).
        file_content: str (content of the file to be created).
    """
    basename = os.path.basename(file_name)
    dir_file = '/'.join(file_name.split('/')[:-1])
    dir_output = dir_file
    try:
        if not os.path.exists(dir_output):
            os.makedirs(dir_output)

        with open(f"{dir_output}/{basename}", mode='w', encoding='utf-8') as file:
            file.write(file_content)
        return True
    except Exception as e:
        logger.error(
            f"Error when creating file {basename} in directory {dir_output}")
        logger.error(e)

    return False
