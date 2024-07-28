from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from utils import read_documents, save_faiss_index, load_faiss_index

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load model and tokenizer
model_name = "t5-small"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize Sentence Transformers model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Paths
dataset_folder = 'dataset/'
faiss_folder = 'faiss_index/'

# Load documents
documents = read_documents(dataset_folder)
document_embeddings = sentence_model.encode(documents)
dim = document_embeddings.shape[1]  # Dimension of embeddings

# Load or create FAISS index
index = load_faiss_index(faiss_folder)
if index is None:
    print("Creating FAISS index...")
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(document_embeddings).astype('float32'))
    save_faiss_index(index, faiss_folder)
    print("FAISS index created and saved.")
else:
    print("FAISS index loaded from existing file.")

# Verify index type
print(f"Index type: {type(index)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    query_embedding = sentence_model.encode([user_input])
    
    # Debugging output
    print(f"Query embedding shape: {query_embedding.shape}")
    print(f"Index type before search: {type(index)}")

    try:
        _, I = index.search(np.array(query_embedding).astype('float32'), k=1)
        context = documents[I[0][0]]

        inputs = tokenizer.encode("generate response: " + context + " " + user_input, return_tensors="pt")
        outputs = model.generate(inputs, max_length=150, num_beams=5, early_stopping=True)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
