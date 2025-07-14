from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from dotenv import load_dotenv

load_dotenv()

# https://python.langchain.com/docs/tutorials/rag/
# https://python.langchain.com/docs/how_to/#document-loaders

'''
web_loader = WebBaseLoader(
    "https://angular.dev/context/llm-files/llms-full.txt")
documentation = web_loader.load()
assert len(documentation) == 1
print(f"Total characters: {len(documentation[0].page_content)}")
'''
try:
    markdown_path = './angular-llm.txt'
    loader = UnstructuredMarkdownLoader(markdown_path)
    data = loader.load()
    assert len(data) == 1
    assert isinstance(data[0], Document)
    readme_content = data[0].page_content

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )
    all_splits = text_splitter.split_documents(data)

    print(f"Split blog post into {len(all_splits)} sub-documents.")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # dbname=supermarket_data user=postgres host=localhost password=p4ssw0rd
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name="angular_docs",
        connection="postgresql+psycopg://postgres:p4ssw0rd@localhost:5432/docs"
    )

    document_ids = vector_store.add_documents(documents=all_splits)

except Exception as e:
    print("Error")
    print(e)
