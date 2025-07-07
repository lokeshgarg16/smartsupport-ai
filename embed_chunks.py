import os
import json
from sentence_transformers import SentenceTransformer
import chromadb

with open("faq_chunks.json","r") as f:
    chunks=json.load(f)

model=SentenceTransformer("all-MiniLM-L6-v2")
embeddings=model.encode(chunks)

client = chromadb.Client()
collection=client.create_collection(name='airtel_faqs')

#  Add data to collection
for i, (text, embedding) in enumerate(zip(chunks, embeddings)):
    collection.add(
        documents=[text],
        embeddings=[embedding],
        ids=[f"faq-{i}"]
    )

print("✅ Embeddings stored in ChromaDB")