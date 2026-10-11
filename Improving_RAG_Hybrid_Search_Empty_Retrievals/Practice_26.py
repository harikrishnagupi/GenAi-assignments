import re
from rank_bm25 import BM25Okapi
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
""",
    "ticket_tracker.txt": """Ticket Tracker Notes

The suspicious login alert is filed under ticket code TKT-4521 and is currently our highest priority case.

The expired certificate warning is filed under ticket code TKT-7788 and is scheduled for renewal within three
business days.

Any follow-up must quote the exact ticket code so the security team can find the right record quickly.
""",
}

splitter = RecursiveCharacterTextSplitter(chunk_size=240, chunk_overlap=30, separators=["\n\n", "\n", ". ", " ", ""])
embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chromadb.Client().create_collection(
    name="secops_chunks_hybrid", embedding_function=embedder, metadata={"hnsw:space": "cosine"}
)

chunk_texts, chunk_meta, bm25 = [], [], None

def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())          # "TKT-9034" -> ["tkt", "9034"]

def add_document(source, text):
    global bm25
    chunks = splitter.split_text(text)
    meta = [{"source": source, "chunk_index": i} for i in range(len(chunks))]
    collection.add(documents=chunks, metadatas=meta, ids=[f"{source}::{i}" for i in range(len(chunks))])
    chunk_texts.extend(chunks)
    chunk_meta.extend(meta)
    bm25 = BM25Okapi([tokenize(c) for c in chunk_texts])   # must be rebuilt after every addition
    return len(chunks)

for _src, _txt in documents.items():
    add_document(_src, _txt)
print("Total chunks stored:", collection.count())

STOPWORDS = {"a", "an", "the", "is", "are", "was", "were", "be", "do", "does", "did", "what", "whats", "which",
             "who", "how", "why", "when", "where", "can", "could", "would", "should", "you", "your", "me", "my", "i",
             "we", "our", "it", "its", "this", "that", "of", "for", "to", "in", "on", "at", "by", "with", "about",
             "and", "or", "please", "tell", "s"}

def semantic_search(query, n_results=5):
    res = collection.query(query_texts=[query], n_results=min(n_results, collection.count()),
                           include=["documents", "metadatas", "distances"])
    return [{"text": t, "source": m["source"], "similarity": round(1 - dist, 4)}
            for t, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]

def keyword_search(query, n_results=5, drop_stopwords=False):
    tokens = [t for t in tokenize(query) if not (drop_stopwords and t in STOPWORDS)]
    if not tokens:
        return []
    scores = bm25.get_scores(tokens)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]
    return [{"text": chunk_texts[i], "source": chunk_meta[i]["source"], "bm25": round(float(scores[i]), 4)}
            for i in order if scores[i] > 0]

def hybrid_search(query, n_results=3, k=60, floor=-1.0, drop_stopwords=False):
    sem = [r for r in semantic_search(query, 10) if r["similarity"] >= floor]
    kw = keyword_search(query, 10, drop_stopwords)
    fused, info = {}, {}
    for ranking in (sem, kw):
        for pos, r in enumerate(ranking):
            fused[r["text"]] = fused.get(r["text"], 0) + 1 / (k + pos + 1)
            info.setdefault(r["text"], r)
    best = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:n_results]
    return [{"text": t, "source": info[t]["source"], "rrf": round(s, 5)} for t, s in best]

def semantic_above(query, n_results=4, threshold=0.5):
    return [r for r in semantic_search(query, n_results) if r["similarity"] >= threshold]

def smart_retrieve(query, n_results=4, threshold=0.5):
    hits = semantic_above(query, n_results, threshold)
    if hits:
        return {"status": "ok", "method": "semantic", "chunks": hits}
    hits = hybrid_search(query, n_results)
    if hits:
        return {"status": "ok", "method": "hybrid_fallback", "chunks": hits}
    topics = ", ".join(sorted({m["source"] for m in chunk_meta}))
    return {"status": "empty", "method": None, "chunks": [],
            "message": f"I don't know based on the provided documents. Covered files: {topics}."}

new_text = """Ticket Addendum

The unpatched file server finding is filed under ticket code TKT-9034 and carries a high severity rating.

The open guest wifi report is filed under ticket code TKT-2210 and is waiting for network team review.
"""
added = add_document("ticket_extra.txt", new_text)
print(f"Added {added} chunks. Total now: {collection.count()}")

def position_of(results, needle):
    for i, r in enumerate(results, 1):
        if needle in r["text"]:
            return i
    return None

query = "TKT-9034"
total = collection.count()
print(f"\nRank of the TKT-9034 chunk for query '{query}':")
print("  semantic only :", position_of(semantic_search(query, total), "TKT-9034"))
print("  keyword (BM25):", position_of(keyword_search(query, total), "TKT-9034"))
print("  hybrid (RRF)  :", position_of(hybrid_search(query, total), "TKT-9034"))

res = smart_retrieve(query, threshold=0.99)      # impossible threshold forces the fallback
print("\nsmart_retrieve ->", res["status"], "|", res["method"])
print("TKT-9034 found through fallback:", any("TKT-9034" in c["text"] for c in res["chunks"]))
