from sentence_transformers import SentenceTransformer
from huggingface_hub import login
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

# Replace 'your_huggingface_token' with your actual Hugging Face token
token = "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"
login(token)

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# embed_function = lambda texts: embedder.encode(texts, convert_to_numpy=True)

def embed_function(texts: str) -> list[float]:
    # print(texts)
    """
    Embed a list of texts using SentenceTransformer.
    
    Args:
        texts: List of strings to embed
        
    Returns:
        numpy.ndarray: Array of embedding vectors
    """
    # Encode all texts at once and return as numpy array
    embeddings = embedder.encode(texts, convert_to_numpy=True)
    # convert to list
    embeddings = embeddings.tolist()
    return embeddings
