# Effect of chunk_overlap (0, 40, 80) at a fixed chunk_size of 200
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

for overlap in (0, 40, 80):
    sp = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=overlap, separators=["\n\n", "\n", ". ", " ", ""])
    chunks = sp.split_text(article_text)
    total = sum(len(c) for c in chunks)
    print(f"overlap={overlap:<3} chunks={len(chunks):<3} total chars stored={total} (original={len(article_text)})")
    if len(chunks) > 1:
        tail = chunks[0][-30:].replace("\n", " ")
        print(f"   end of chunk 0  : ...{tail!r}")
        print(f"   start of chunk 1: {chunks[1][:30]!r}...")
