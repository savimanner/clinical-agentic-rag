import os
import chromadb
from dotenv import load_dotenv
import requests

load_dotenv()

client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_collection(
    name="guidelines"
)



response = requests.post(
  "https://openrouter.ai/api/v1/embeddings",
  headers={
    "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
    "Content-Type": "application/json",
  },
  json={
    "model": f"{os.environ['OPENROUTER_EMBEDDING_MODEL']}",
    "input": "Alaseljavalu ravi"
  }
)

data = response.json()
embedding = data["data"][0]["embedding"]

results = collection.query(
    query_embeddings=[embedding],
    n_results=1
)

print(results["metadatas"][0][0]["search_text"])
