import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

documents = {
    "network_security.txt": """Network Security Essentials

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
""",
    "encryption_basics.txt": """Encryption Basics

Symmetric encryption uses one shared key for both locking and unlocking data, which makes it fast. Asymmetric
encryption uses a public key and a private key, which makes safe key exchange possible.

Hashing turns data into a fixed-length fingerprint. Good password storage uses salted hashes so identical
passwords never produce identical stored values.
"""
}

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=25, separators=["\n\n", "\n", ". ", " ", ""])
embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chromadb.Client().create_collection(
    name="secops_chunks", embedding_function=embedder, metadata={"hnsw:space": "cosine"}
)

def index_document(source, text):
    chunks = splitter.split_text(text)
    collection.add(
        documents=chunks,
        metadatas=[{"source": source, "chunk_index": i} for i in range(len(chunks))],
        ids=[f"{source}_chunk_{i}" for i in range(len(chunks))],
    )

for _source, _text in documents.items():
    index_document(_source, _text)
print("Total chunks stored:", collection.count())

def to_similarity(distance):
    return 1 - distance

def fetch_chunks(query, n_results=5, min_sim=0.25, source=None, where=None, max_chars=None):
    kwargs = dict(query_texts=[query], n_results=n_results, include=["documents", "metadatas", "distances"])
    if where:
        kwargs["where"] = where
    elif source:
        kwargs["where"] = {"source": source}
    res = collection.query(**kwargs)

    hits, used = [], 0
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        sim = to_similarity(dist)
        if sim < min_sim:
            continue
        if max_chars is not None and used + len(doc) > max_chars:
            break
        used += len(doc)
        hits.append({"text": doc, "source": meta["source"], "chunk_index": meta["chunk_index"], "similarity": round(sim, 4)})
    return hits

def best_similarity(q):
    hits = fetch_chunks(q, n_results=5, min_sim=-1.0)      # keep everything, just measure
    return max(h["similarity"] for h in hits)

in_scope = [best_similarity(q) for q in ['What is phishing?', 'Why is multi-factor authentication useful?', 'How does hashing protect passwords?']]
out_scope = [best_similarity(q) for q in ["What is the capital of Australia?", "Who won the football world cup?"]]
print("Best similarity, in-scope queries    :", in_scope)
print("Best similarity, out-of-scope queries:", out_scope)

suggested = (min(in_scope) + max(out_scope)) / 2
print("Suggested threshold (midpoint):", round(suggested, 3))
print("Chunks kept for the out-of-scope question at that threshold:", len(fetch_chunks("What is the capital of Australia?", min_sim=suggested)))
