# data_preparation.py
import json

documents = [
    {"id": 1, "text": "Python is a programming language that lets you work quickly and integrate systems more effectively."},
    {"id": 2, "text": "Machine learning is the study of computer algorithms that improve automatically through experience."},
    {"id": 3, "text": "Flask is a micro web framework written in Python."},
]

with open('documents.json', 'w') as f:
    json.dump(documents, f)
