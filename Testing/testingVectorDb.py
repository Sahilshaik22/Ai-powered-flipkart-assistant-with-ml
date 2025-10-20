from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

root = Path(__file__).resolve().parents[1]
vector_dir = root/"VectorDb"

embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory=str(vector_dir),embedding_function=embeddings,collection_name="Filpkart_Polices")

query = "what is return policy for damged items?"

result = db.similarity_search(query)

for i,doc in enumerate(result,1):
    print(doc)
    


