from pathway.xpacks.llm.vector_store import VectorStoreClient

# Initialize the client
client = VectorStoreClient(host="127.0.0.1", port=8000)

# Define your query text
query_text = "equations regarding Wasserstein distance"
# Perform the query, retrieving the top 5 nearest neighbors
results = client.query(query=query_text, k=30)

# Print the results
with open("work.txt", "w") as f:
    for idx, result in enumerate(results, start=1):
        f.write(f"Result {idx}: {result}\n")