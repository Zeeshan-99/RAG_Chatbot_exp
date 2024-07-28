# create_index.py
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load documents
with open('documents.json', 'r') as f:
    documents = json.load(f)

# Load pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed the documents
corpus_embeddings = model.encode([doc['text'] for doc in documents])

# Convert to float32 as FAISS requires this
corpus_embeddings = np.array(corpus_embeddings, dtype=np.float32)

# Create FAISS index
dimension = corpus_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(corpus_embeddings)

# Save the index
faiss.write_index(index, 'document_index.faiss')
