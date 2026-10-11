from Practice_28 import retrieve_context, build_user_prompt, ask_llm

question = "How do encryption and firewalls work together to protect data?"
for n in (5, 2):
    chunks = retrieve_context(question, n_results=n)
    prompt = build_user_prompt(question, chunks)
    print(f"\nn_results={n}: prompt length = {len(prompt)} characters, chunks = {len(chunks)}")
    print("Covered sources:", sorted({c['source'] for c in chunks}))
    print("Answer:", ask_llm(prompt)[:250].replace("\n", " "))
