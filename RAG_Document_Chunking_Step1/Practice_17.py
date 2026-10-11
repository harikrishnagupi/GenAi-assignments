# 3 sentences per chunk with 1 sentence of overlap
import re

article_text = """Network security protects data as it moves between devices. Firewalls filter traffic using rules, while
intrusion detection systems watch for patterns that look like an attack.

A virtual private network, or VPN, creates an encrypted tunnel between a device and a remote network. It is
widely used so that employees can reach internal systems safely from outside the office.

Phishing is one of the most common attack methods. Attackers send messages that look genuine to trick people
into sharing passwords or opening harmful attachments.

Multi-factor authentication adds a second proof of identity, such as a code on a phone. Even if a password is
stolen, the attacker still cannot log in without the second factor.

Regular software updates close known vulnerabilities. Many large breaches happen because a patch that was
available for months was never installed."""

def split_sentences(text):
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

def overlapping_chunks(text, size=3, overlap=1):
    s = split_sentences(text)
    step = size - overlap
    out = []
    for i in range(0, len(s), step):
        out.append(s[i:i + size])
        if i + size >= len(s):
            break
    return out

for idx, group in enumerate(overlapping_chunks(article_text)):
    print(f"Chunk {idx}: {len(group)} sentences")
    print("   first :", group[0][:60])
    print("   last  :", group[-1][:60])
