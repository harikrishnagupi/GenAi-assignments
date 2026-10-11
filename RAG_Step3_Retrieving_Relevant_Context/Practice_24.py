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

third_doc = """Incident Response

An incident response plan describes who does what when a breach happens. The usual phases are preparation,
detection, containment, eradication, and recovery.

After the incident, a blameless review records what went wrong and how to prevent it. Sharing the lessons
learned helps the whole team improve its defences.
"""
new_source = "incident_response.txt"
index_document(new_source, third_doc)
print("Total chunks now:", collection.count())

wanted = ["network_security.txt", new_source]
hits = fetch_chunks("Why do teams run a blameless review?", n_results=6, min_sim=-1.0, where={"source": {"$in": wanted}})
print("Searching only in:", wanted)
for h in hits:
    print(f"   [{h['similarity']}] {h['source']} #{h['chunk_index']}")

other = fetch_chunks("Why do teams run a blameless review?", n_results=6, min_sim=-1.0, where={"source": {"$ne": new_source}})
print("\nWith the new document excluded:", sorted({h['source'] for h in other}))
