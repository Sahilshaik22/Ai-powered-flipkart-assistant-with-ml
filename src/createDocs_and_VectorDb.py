from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
from datetime import date
load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

root = Path(__file__).resolve().parents[1]
pdf_path = root/"Data"/"Flipkart_policies.pdf"
vector_dir = root/"VectorDb"

def main():
    if not pdf_path.exists():
        raise FileNotFoundError(f"Pdf not found:{pdf_path}")
    print("Fetching Documents.................")
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()
    
    print("Loaded pages from Pdf")
    print("Starting spliiting  Docusments")
    
    for p in docs:
        p.metadata.update({
            "brand":"Flipkart",
            "doc_name": pdf_path.name,
            "country":"India",
            "page":p.metadata.get("page"),
            "source":str(pdf_path),
            "updated_at":date.today().isoformat()
        
        })

    text_splitters = RecursiveCharacterTextSplitter(chunk_size = 800,chunk_overlap = 150)
    chunks = text_splitters.split_documents(docs)

    print(f"Created {len(chunks)} text Chunks")
    print("creating Embeddings........ ")
    
    embeddings = HuggingFaceEmbeddings(model_name= "sentence-transformers/all-MiniLM-L6-v2")
    
    vectordb = Chroma.from_documents(embedding=embeddings,documents=chunks,persist_directory=str(vector_dir),collection_name="Filpkart_Polices")
    vectordb.persist()

    print("Congratualtions VectorDb is ready")


if  __name__ == "__main__":
    main()


    
        