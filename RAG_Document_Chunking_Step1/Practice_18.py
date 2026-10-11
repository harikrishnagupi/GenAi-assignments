import os
import re
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

file_path = "my_security_notes.txt"
if not os.path.exists(file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("""Encryption Basics

Symmetric encryption uses one shared key for both locking and unlocking data, which makes it fast. Asymmetric
encryption uses a public key and a private key, which makes safe key exchange possible.

Hashing turns data into a fixed-length fingerprint. Good password storage uses salted hashes so identical
passwords never produce identical stored values.

Incident Response

An incident response plan describes who does what when a breach happens. The usual phases are preparation,
detection, containment, eradication, and recovery.

After the incident, a blameless review records what went wrong and how to prevent it. Sharing the lessons
learned helps the whole team improve its defences.""")

with open(file_path, "r", encoding="utf-8") as f:
    my_text = f.read()
print("Loaded", file_path, "->", len(my_text), "characters")

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    encode, decode = _enc.encode, _enc.decode
except Exception:                      # offline fallback: treat words as tokens
    encode = lambda t: t.split(" ")
    decode = lambda toks: " ".join(toks)

def fixed_chunks(text, size=350, overlap=70):
    out, start = [], 0
    while start < len(text):
        out.append(text[start:start + size])
        start += size - overlap
    return out

def sentence_chunks(text, n=4):
    s = [x for x in re.split(r'(?<=[.!?])\s+', text.strip()) if x]
    return [" ".join(s[i:i + n]) for i in range(0, len(s), n)]

def token_chunks(text, max_tokens=70, overlap=14):
    toks = encode(text)
    out, start = [], 0
    while start < len(toks):
        out.append(decode(toks[start:start + max_tokens]))
        start += max_tokens - overlap
    return out

strategies = {
    "fixed-size": fixed_chunks(my_text),
    "character": CharacterTextSplitter(separator="\n\n", chunk_size=350, chunk_overlap=70).split_text(my_text),
    "recursive": RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=70, separators=["\n\n", "\n", ". ", " ", ""]).split_text(my_text),
    "sentence": sentence_chunks(my_text),
    "token": token_chunks(my_text),
}

def mid_word_cuts(chunks):
    return sum(1 for c in chunks if c and c[0].isalpha() and c[0].islower())

print("Chunks that start in the middle of a word/sentence (lowercase start):")
for name, ch in strategies.items():
    print(f"  {name:<12} {mid_word_cuts(ch):>2} of {len(ch)}")
