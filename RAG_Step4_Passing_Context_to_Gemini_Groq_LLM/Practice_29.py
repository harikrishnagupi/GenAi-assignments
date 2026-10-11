from Practice_28 import collection, store_document, rag_answer, retrieve_context

before = collection.count()
store_document("incident_response.txt", """Incident Response

An incident response plan describes who does what when a breach happens. The usual phases are preparation,
detection, containment, eradication, and recovery.

After the incident, a blameless review records what went wrong and how to prevent it. Sharing the lessons
learned helps the whole team improve its defences.
""")
print(f"Chunks: {before} -> {collection.count()}")

for q in ['What are the phases of incident response?', 'Why do teams run a blameless review?', 'What is the capital of Australia?']:
    found = retrieve_context(q)
    print("\nQuestion:", q)
    print("  retrieved:", [(c["source"], c["similarity"]) for c in found] or "nothing above the threshold")
    print("  answer   :", rag_answer(q, show_sources=False))
