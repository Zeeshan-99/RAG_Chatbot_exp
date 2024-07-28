import os
import faiss

def save_faiss_index(index, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    faiss.write_index(index, os.path.join(folder_path, 'index.faiss'))

def load_faiss_index(folder_path):
    index_file = os.path.join(folder_path, 'index.faiss')
    if os.path.exists(index_file):
        index = faiss.read_index(index_file)
        print(f"Loaded FAISS index from {index_file}")
        return index
    else:
        print(f"No FAISS index found at {index_file}")
        return None
