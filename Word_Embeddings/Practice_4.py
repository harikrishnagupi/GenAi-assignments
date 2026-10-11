# One-hot encoding and what happens with a word the encoder has never seen
from sklearn.preprocessing import OneHotEncoder
import numpy as np

known = ['firewall', 'router', 'switch', 'firewall', 'modem', 'proxy', 'router', 'gateway']
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoder.fit([[w] for w in known])

print("Known categories:", list(encoder.categories_[0]))

unseen = [["zebra"]]
print("Unseen word 'zebra' ->", encoder.transform(unseen)[0])   # all zeros, no crash

vecs = encoder.transform([[encoder.categories_[0][0]], [encoder.categories_[0][1]]])
print("Dot product of two different words:", float(np.dot(vecs[0], vecs[1])))
print("Dot product of a word with itself :", float(np.dot(vecs[0], vecs[0])))
