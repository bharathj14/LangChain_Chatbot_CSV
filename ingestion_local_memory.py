from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
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

    print("Ingesting....")
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(os.environ["FAISS_INDEX"])
    print("****Loading to InMemory Completed ***")





if __name__ == "__main__":
    ingest_docs()