import os
import hashlib
import tempfile
from collections import Counter
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

first_text = """Network Security Essentials

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
second_text = """Encryption Basics

Symmetric encryption uses one shared key for both locking and unlocking data, which makes it fast. Asymmetric
encryption uses a public key and a private key, which makes safe key exchange possible.

Hashing turns data into a fixed-length fingerprint. Good password storage uses salted hashes so identical
passwords never produce identical stored values.
"""
third_text = """Incident Response

An incident response plan describes who does what when a breach happens. The usual phases are preparation,
detection, containment, eradication, and recovery.

After the incident, a blameless review records what went wrong and how to prevent it. Sharing the lessons
learned helps the whole team improve its defences.
"""
sources = {"network_security.txt": first_text, "encryption_basics.txt": second_text, "incident_response.txt": third_text}

os.makedirs("docs_security", exist_ok=True)
for name, text in sources.items():
    with open(os.path.join("docs_security", name), "w", encoding="utf-8") as f:
        f.write(text)

splitter = RecursiveCharacterTextSplitter(chunk_size=260, chunk_overlap=30, separators=["\n\n", "\n", ". ", " ", ""])
embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

db_dir = os.path.join(tempfile.gettempdir(), "security_chroma_store")
client = chromadb.PersistentClient(path=db_dir)
collection = client.get_or_create_collection(name="secops_chunks", embedding_function=embedder)

if collection.count() == 0:
    for name, text in sources.items():
        pieces = splitter.split_text(text)
        collection.add(
            documents=pieces,
            metadatas=[{"source": name, "chunk_index": i} for i in range(len(pieces))],
            ids=[f"{name}_{i}" for i in range(len(pieces))],
        )
print("Chunks stored on disk:", collection.count())

# Simulate a restart: open a brand new client on the same folder
del collection, client
reopened = chromadb.PersistentClient(path=db_dir).get_collection(name="secops_chunks", embedding_function=embedder)
print("Chunks after reopening :", reopened.count())

res = reopened.query(query_texts=["Why do salted hashes matter for passwords?"], n_results=3, include=["documents", "metadatas", "distances"])
for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
    print(f"[{dist:.3f}] {meta['source']} (chunk {meta['chunk_index']})")
print("Database folder:", db_dir)
