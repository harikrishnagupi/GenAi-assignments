# Effect of min_count on the vocabulary
from gensim.models import Word2Vec

data = [['firewall', 'blocks', 'unwanted', 'traffic'], ['proxy', 'filters', 'unwanted', 'traffic'], ['router', 'forwards', 'network', 'packets'], ['switch', 'forwards', 'network', 'frames'], ['vpn', 'encrypts', 'network', 'traffic'], ['tls', 'encrypts', 'web', 'traffic'], ['phishing', 'is', 'a', 'common', 'attack'], ['malware', 'is', 'a', 'common', 'attack'], ['router', 'connects', 'networks'], ['gateway', 'connects', 'networks']]

for mc in (1, 2):
    m = Word2Vec(data, vector_size=10, window=2, min_count=mc, workers=1, seed=7, epochs=30)
    print(f"min_count={mc}: vocabulary size = {len(m.wv)}")
    print("   words kept:", sorted(m.wv.index_to_key))
    print("   neighbours of 'router':", [w for w, _ in m.wv.most_similar("router", topn=3)])
