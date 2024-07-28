# app.py
from flask import Flask, request, jsonify, render_template
import json
from sentence_transformers import SentenceTransformer
import faiss
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# Load documents and FAISS index
with open('documents.json', 'r') as f:
    documents = json.load(f)

index = faiss.read_index('document_index.faiss')

# Load models
embedder = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '')
    if not user_input:
        return jsonify({'error': 'No message provided'}), 400

    user_embedding = embedder.encode([user_input], convert_to_tensor=True).detach().cpu().numpy()

    # Perform similarity search
    _, top_k_indices = index.search(user_embedding, k=1)
    best_doc = documents[top_k_indices[0][0]]['text']

    # Generate response using GPT-2
    input_text = f"Context: {best_doc}\nUser: {user_input}\nBot:"
    inputs = tokenizer.encode(input_text, return_tensors='pt')
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({'response': response.split("Bot:")[-1].strip()})

if __name__ == '__main__':
    app.run(debug=True)
