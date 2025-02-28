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
    # Load CSV using CSVLoader (Assumes first row is the header)
    loader = CSVLoader(file_path="./csv/KEDB.csv", encoding="utf-8")
    raw_documents = loader.load()

    print(f"Loaded {len(raw_documents)} documents")

    documents = []
    for doc in raw_documents:
        metadata = doc.metadata  # This will include the headers from the CSV
        content_dict = doc.to_dict()  # Convert to dictionary format

        try:
            # Extracting values from CSV
            number = content_dict["number"]
            version = content_dict["version"]
            short_desc = content_dict["short description"]
            text = content_dict["text"]
            author = content_dict["author"]
            kb_category = content_dict["kb_category"]

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
    vectorstore.save_local("faiss_index_react")

    print("**** Loading to vector store done ****")

    # Load vector store
    new_vectorstore = FAISS.load_local(
        "faiss_index_react",
        embeddings,
        allow_dangerous_deserialization=True
    )


if __name__ == "__main__":
    ingest_docs()
