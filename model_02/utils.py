import fitz  # PyMuPDF
import docx
import json
import os
import faiss

def read_pdf(file_path):
    text = ""
    doc = fitz.open(file_path)
    for page in doc:
        text += page.get_text()
    return text

def read_docx(file_path):
    doc = docx.Document(file_path)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return text

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return json.dumps(data)

def read_documents(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith('.pdf'):
            documents.append(read_pdf(file_path))
        elif filename.endswith('.docx'):
            documents.append(read_docx(file_path))
        elif filename.endswith('.json'):
            documents.append(read_json(file_path))
    return documents

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
