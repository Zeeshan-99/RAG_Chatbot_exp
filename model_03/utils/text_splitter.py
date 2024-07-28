# from typing import List

# def split_text_into_chunks(documents: List[dict], chunk_size: int = 512) -> List[dict]:
#     chunks = []
#     for doc in documents:
#         text = doc['text']
#         for i in range(0, len(text), chunk_size):
#             chunk = text[i:i + chunk_size]
#             chunks.append({'text': chunk, 'source': doc['type']})
#     return chunks
def split_text(text, chunk_size=1000, overlap=100):
    """Splits text into chunks of a specified size with a specified overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end - overlap
    return chunks
