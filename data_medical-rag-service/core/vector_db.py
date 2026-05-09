import chromadb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "vectordb" / "chroma"

client = chromadb.PersistentClient(path=str(DB_PATH))
collection = client.get_or_create_collection(name="medical_kb")