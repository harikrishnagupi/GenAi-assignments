# Rank every word in the vocabulary by similarity to one target word
from gensim.models import Word2Vec

sentences = [['firewall', 'blocks', 'unwanted', 'traffic'], ['proxy', 'filters', 'unwanted', 'traffic'], ['router', 'forwards', 'network', 'packets'], ['switch', 'forwards', 'network', 'frames'], ['vpn', 'encrypts', 'network', 'traffic'], ['tls', 'encrypts', 'web', 'traffic'], ['phishing', 'is', 'a', 'common', 'attack'], ['malware', 'is', 'a', 'common', 'attack'], ['router', 'connects', 'networks'], ['gateway', 'connects', 'networks']]
model = Word2Vec(sentences, vector_size=10, window=2, min_count=1, workers=1, seed=21, epochs=100)

target = "router"
scores = sorted(
    ((w, float(model.wv.similarity(target, w))) for w in model.wv.index_to_key if w != target),
    key=lambda x: x[1], reverse=True,
)
print(f"Words ranked by similarity to '{target}':")
for w, s in scores[:8]:
    bar = "#" * max(0, int(s * 20))
    print(f"{w:<12} {s:6.3f} {bar}")
