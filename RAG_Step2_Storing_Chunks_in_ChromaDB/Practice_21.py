import hashlib
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

def add_unique(collection, splitter, text, source):
    """Store only chunks whose content hash is new. Returns (added, skipped)."""
    chunks = splitter.split_text(text)
    existing = set(collection.get()["ids"])
    new_docs, new_ids, new_meta = [], [], []
    for i, ch in enumerate(chunks):
        cid = hashlib.sha1(ch.encode("utf-8")).hexdigest()[:10]
        if cid in existing or cid in new_ids:
            continue
        new_docs.append(ch); new_ids.append(cid)
        new_meta.append({"source": source, "chunk_index": i})
    if new_docs:
        collection.add(documents=new_docs, ids=new_ids, metadatas=new_meta)
    return len(new_docs), len(chunks) - len(new_docs)

embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chromadb.Client().create_collection(name="secops_chunks", embedding_function=embedder)
splitter = RecursiveCharacterTextSplitter(chunk_size=220, chunk_overlap=20, separators=["\n\n", "\n", ". ", " ", ""])

doc_a = """Encryption Basics

Symmetric encryption uses one shared key for both locking and unlocking data, which makes it fast. Asymmetric
encryption uses a public key and a private key, which makes safe key exchange possible.

Hashing turns data into a fixed-length fingerprint. Good password storage uses salted hashes so identical
passwords never produce identical stored values.
"""

doc_b = """Incident Response

An incident response plan describes who does what when a breach happens. The usual phases are preparation,
detection, containment, eradication, and recovery.

After the incident, a blameless review records what went wrong and how to prevent it. Sharing the lessons
learned helps the whole team improve its defences.
"""

doc_c = """Ticket Tracker Notes

The suspicious login alert is filed under ticket code TKT-4521 and is currently our highest priority case.

The expired certificate warning is filed under ticket code TKT-7788 and is scheduled for renewal within three
business days.

Any follow-up must quote the exact ticket code so the security team can find the right record quickly.
"""

print("First store :", add_unique(collection, splitter, doc_a, "encryption_basics.txt"))
print("Same again  :", add_unique(collection, splitter, doc_a, "encryption_basics.txt"))     # everything skipped
print("Second doc  :", add_unique(collection, splitter, doc_b, "incident_response.txt"))
print("Third doc   :", add_unique(collection, splitter, doc_c, "ticket_tracker.txt"))
print("Total chunks:", collection.count())
top = collection.query(query_texts=["What are the phases of incident response?"], n_results=1, include=["documents", "metadatas"])
print("Top hit:", top["metadatas"][0][0]["source"], "|", top["documents"][0][0][:70].replace("\n", " "))
