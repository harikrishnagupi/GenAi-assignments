import re
from Practice_28 import rag_answer

NUMBERED = (
    "Use ONLY the provided context. If the answer is missing, reply exactly: "
    "\"I don't know based on the provided documents.\" "
    "Otherwise write a numbered list of exactly 2 steps or facts ('1.' and '2.'), one sentence each, "
    "then a last line beginning with 'Source:'."
)

for q in ['What is phishing?', 'How do encryption and firewalls work together to protect data?']:
    reply = rag_answer(q, system_prompt=NUMBERED, show_sources=False)
    items = re.findall(r"^\s*\d+\.", reply, flags=re.MULTILINE)
    has_source = any(line.strip().lower().startswith("source") for line in reply.splitlines())
    print(f"\n{q}\n{reply}\n-> numbered items: {len(items)}, source line present: {has_source}")
