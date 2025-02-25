from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
import os

load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import  CSVLoader
from langchain_pinecone import PineconeVectorStore


embeddings = OllamaEmbeddings(
    model="mistral",
    temperature=0
)
def ingest_docs():
    loader = CSVLoader(file_path="Path")

    raw_documents = loader.load()
    print(raw_documents)
    print(f"loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)


    print(f"Going to add {len(documents)} to Pinecone")
    PineconeVectorStore.from_documents(
        documents, embeddings, index_name=os.environ["INDEX_NAME"]
    )
    print("****Loading to vectorstore done ***")


if __name__ == "__main__":
    ingest_docs()