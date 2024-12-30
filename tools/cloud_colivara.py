# Import and initialize ColiVara
from colivara_py import ColiVara

rag_client = ColiVara()

# Upload a document to the default collection
document = rag_client.upsert_document(
    name="sample_document",
    url="https://example.com/sample.pdf",
    metadata={"author": "John Doe"},
)
results = rag_client.search(query="machine learning")
print(results)  # top 3 pages with the most relevant information
