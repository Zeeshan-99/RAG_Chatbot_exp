# from flask import Blueprint, request, jsonify, render_template
# from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
# from langchain.vectorstores import FAISS
# from langchain.embeddings import SentenceTransformerEmbeddings
# from utils.document_reader import read_documents
# from utils.faiss_handler import load_faiss_index, save_faiss_index
# from utils.text_splitter import split_text
# import numpy as np

# main = Blueprint('main', __name__)

# # Load model and tokenizer from local directory
# # model_dir = "models/model"
# # model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
# # tokenizer = AutoTokenizer.from_pretrained(model_dir)
# # llm = pipeline('text2text-generation', model=model, tokenizer=tokenizer)
# #######--------------------------------------------------
# ### Load model and tokenizer from Hugging Face
# model_name = "t5-small"
# llm = pipeline('text2text-generation', model=model_name)
# #######--------------------------------------------------

# # Initialize Sentence Transformers model
# embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# # Paths
# dataset_folder = 'dataset/'
# faiss_folder = 'faiss_index/'

# # Load documents and create vector store
# documents = read_documents(dataset_folder)
# document_texts = [doc['text'] for doc in documents]
# split_texts = [split_text(text) for text in document_texts]
# flattened_texts = [chunk for text in split_texts for chunk in text]

# # Load or create FAISS index
# index = load_faiss_index(faiss_folder)
# if index is None:
#     print("Creating FAISS index...")
#     index = FAISS.from_texts(flattened_texts, embeddings)
#     save_faiss_index(index, faiss_folder)
#     print("FAISS index created and saved.")
# else:
#     print("FAISS index loaded from existing file.")

# @main.route('/')
# def index():
#     return render_template('index.html')

# @main.route('/chat', methods=['POST'])
# def chat():
#     user_input = request.json.get('message')
#     query_embedding = embeddings.embed(user_input)
    
#     try:
#         _, I = index.search(np.array([query_embedding]), k=1)
#         context = flattened_texts[I[0][0]]

#         response = llm(f"Context: {context}\nUser: {user_input}\nBot:")[0]['generated_text']
#         return jsonify({'response': response})
#     except Exception as e:
#         return jsonify({'error': str(e)})

#####--------------------------
from flask import Blueprint, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from utils.document_reader import read_documents
from utils.faiss_handler import load_faiss_index, save_faiss_index
from utils.text_splitter import split_text
import numpy as np

main = Blueprint('main', __name__)

# Load Hugging Face model and tokenizer
model_name = "t5-small"  # Replace with your model directory if it's local
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Initialize Sentence Transformers model
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Paths
dataset_folder = 'dataset/'
faiss_folder = 'faiss_index/'

# Load documents and create vector store
documents = read_documents(dataset_folder)
document_texts = [doc['text'] for doc in documents]
split_texts = [split_text(text) for text in document_texts]
flattened_texts = [chunk for text in split_texts for chunk in text]

# Load or create FAISS index
index = load_faiss_index(faiss_folder)
if index is None:
    print("Creating FAISS index...")
    index = FAISS.from_texts(flattened_texts, embeddings)
    save_faiss_index(index, faiss_folder)
    print("FAISS index created and saved.")
else:
    print("FAISS index loaded from existing file.")

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    query_embedding = embeddings.embed(user_input)
    
    try:
        _, I = index.search(np.array([query_embedding]), k=1)
        context = flattened_texts[I[0][0]]

        # Prepare input for Hugging Face model
        input_text = f"Context: {context}\nUser: {user_input}\nBot:"
        inputs = tokenizer(input_text, return_tensors="pt")
        outputs = model.generate(**inputs)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)})
#########-----------------------------------

# from flask import Blueprint, request, jsonify, render_template
# from langchain.llms import OpenAI
# from langchain.vectorstores import FAISS
# from langchain.embeddings import SentenceTransformerEmbeddings
# from utils.document_reader import read_documents
# from utils.faiss_handler import load_faiss_index, save_faiss_index
# from utils.text_splitter import split_text
# import numpy as np

# main = Blueprint('main', __name__)

# # Load model and tokenizer
# model_name = "t5-small"
# llm = OpenAI(model_name=model_name)

# # Initialize Sentence Transformers model
# embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# # Paths
# dataset_folder = 'dataset/'
# faiss_folder = 'faiss_index/'

# # Load documents and create vector store
# documents = read_documents(dataset_folder)
# document_texts = [doc['text'] for doc in documents]
# split_texts = [split_text(text) for text in document_texts]
# flattened_texts = [chunk for text in split_texts for chunk in text]

# # Load or create FAISS index
# index = load_faiss_index(faiss_folder)
# if index is None:
#     print("Creating FAISS index...")
#     index = FAISS.from_texts(flattened_texts, embeddings)
#     save_faiss_index(index, faiss_folder)
#     print("FAISS index created and saved.")
# else:
#     print("FAISS index loaded from existing file.")

# @main.route('/')
# def index():
#     return render_template('index.html')

# @main.route('/chat', methods=['POST'])
# def chat():
#     user_input = request.json.get('message')
#     query_embedding = embeddings.embed(user_input)
    
#     try:
#         _, I = index.search(np.array([query_embedding]), k=1)
#         context = flattened_texts[I[0][0]]

#         response = llm({"prompt": f"Context: {context}\nUser: {user_input}\nBot:"})
#         return jsonify({'response': response})
#     except Exception as e:
#         return jsonify({'error': str(e)})
