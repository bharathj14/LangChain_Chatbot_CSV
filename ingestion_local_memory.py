from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
import os
from langchain.schema import Document

load_dotenv()

from langchain_community.document_loaders import  CSVLoader



embeddings = OllamaEmbeddings(
    model="mistral",
    temperature=0
)
def ingest_docs():

    loader = CSVLoader(file_path="Path")

    raw_documents = loader.load()
    print(raw_documents)
    print(f"loaded {len(raw_documents)} documents")

    print(f"Loaded {len(raw_documents)} documents")

    documents = []
    for doc in raw_documents:
        metadata = doc.metadata
        csv_content = doc.page_content.split(",")  # Assuming CSVLoader keeps content in page_content

        if len(csv_content) < 6:
            continue  # Skip invalid rows

        number, version, short_desc, text, author, kb_category = csv_content

        # Creating new document with relevant metadata and text content
        new_doc = Document(
            page_content=f"{short_desc} {text}",
            metadata={"source": number, "version": version, "author": author, "kb_category": kb_category}
        )
        documents.append(new_doc)

    print(f"Going to add {len(documents)} documents to vector storage")

    # Store in FAISS vector store
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local("faiss_index_react")

    print("**** Loading to vector store done ****")


if __name__ == "__main__":
    ingest_docs()