from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("sentence-transformers/all-mini-l6-v2")
embed_function = lambda text: embedder.encode([text])[0]