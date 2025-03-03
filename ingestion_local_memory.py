import os

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain.schema import Document

load_dotenv()

embeddings = OllamaEmbeddings(
    model="mistral",
    temperature=0
)


def ingest_docs():
    # Load CSV using CSVLoader
    loader = CSVLoader(file_path=os.environ["FOLDER_PATH"])
    raw_documents = loader.load()

    print(f"Loaded {len(raw_documents)} documents")

    documents = []
    for doc in raw_documents:
        metadata = doc.metadata  # This contains column names from CSV as keys
        page_content = doc.page_content  # Contains raw text of the row

        try:
            # Extracting values from CSV (metadata should contain headers as keys)
            number = metadata.get("number", "").strip()
            version = metadata.get("version", "").strip()
            short_desc = metadata.get("short description", "").strip()
            text = metadata.get("text", "").strip()
            author = metadata.get("author", "").strip()
            kb_category = metadata.get("kb_category", "").strip()

            # Creating new document with relevant metadata and combined text
            new_doc = Document(
                page_content=f"{short_desc} {text}",
                metadata={"source": number, "version": version, "author": author, "kb_category": kb_category}
            )
            documents.append(new_doc)

        except KeyError as e:
            print(f"Skipping row due to missing field: {e}")

    print(f"Going to add {len(documents)} documents to vector storage")

    # Store in FAISS vector store
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(os.environ["FAISS_INDEX"])

    print("**** Loading to vector store done ****")

    # Load vector store
    # new_vectorstore = FAISS.load_local(
    #     "faiss_index_react",
    #     embeddings,
    #     allow_dangerous_deserialization=True
    # )


if __name__ == "__main__":
    ingest_docs()
