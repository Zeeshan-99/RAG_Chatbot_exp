import fitz  # PyMuPDF
import docx
import json
import os

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
            documents.append({'text': read_pdf(file_path), 'type': 'pdf'})
        elif filename.endswith('.docx'):
            documents.append({'text': read_docx(file_path), 'type': 'docx'})
        elif filename.endswith('.json'):
            documents.append({'text': read_json(file_path), 'type': 'json'})
    return documents
