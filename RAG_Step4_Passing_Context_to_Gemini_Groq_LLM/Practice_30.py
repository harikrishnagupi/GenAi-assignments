from Practice_28 import retrieve_context, build_user_prompt, ask_llm
import difflib

question = "Why do salted hashes matter for passwords?"
prompt = build_user_prompt(question, retrieve_context(question))

def consistency(temp):
    a, b = ask_llm(prompt, temperature=temp), ask_llm(prompt, temperature=temp)
    return round(difflib.SequenceMatcher(None, a, b).ratio(), 3), a, b

for temp in (0.1, 1.5):
    ratio, a, b = consistency(temp)
    print(f"temperature={temp}  similarity between two runs = {ratio}")
    print("   run 1:", a[:100].replace("\n", " "))
    print("   run 2:", b[:100].replace("\n", " "))
