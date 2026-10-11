import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

article_text = """Network Security Essentials

Network security protects data as it moves between devices. Firewalls filter traffic using rules, while
intrusion detection systems watch for patterns that look like an attack.

A virtual private network, or VPN, creates an encrypted tunnel between a device and a remote network. It is
widely used so that employees can reach internal systems safely from outside the office.

Phishing is one of the most common attack methods. Attackers send messages that look genuine to trick people
into sharing passwords or opening harmful attachments.

Multi-factor authentication adds a second proof of identity, such as a code on a phone. Even if a password is
stolen, the attacker still cannot log in without the second factor.

Regular software updates close known vulnerabilities. Many large breaches happen because a patch that was
available for months was never installed.
"""

embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.Client()
collection = client.create_collection(name="secops_chunks", embedding_function=embedder)

for overlap in (0, 60):
    sp = RecursiveCharacterTextSplitter(chunk_size=220, chunk_overlap=overlap, separators=["\n\n", "\n", ". ", " ", ""])
    pieces = sp.split_text(article_text)
    collection.add(
        documents=pieces,
        ids=[f"ov{overlap}_{i}" for i in range(len(pieces))],
        metadatas=[{"overlap": overlap, "idx": i} for i in range(len(pieces))],
    )
    print(f"overlap={overlap}: {len(pieces)} chunks")

query = "What is phishing?"
for overlap in (0, 60):
    res = collection.query(query_texts=[query], n_results=2, where={"overlap": overlap}, include=["documents", "distances"])
    print(f"\n[overlap={overlap}] query: {query}")
    for doc, dist in zip(res["documents"][0], res["distances"][0]):
        print(f"   {dist:.4f} {doc[:90]!r}")
