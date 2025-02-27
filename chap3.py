#%%
import sys
sys.path.append('..')
import numpy as np
from common.layers import MatMul
from common.util import preprocess
# %%
c = np.array([1,0,0,0,0,0,0])
W = np.random.randn(7,3)
layer = MatMul(W)
h = layer.forward(c)
print(h)
# %%
c0 = np.array([[1,0,0,0,0,0,0]])
c1 = np.array([[0,0,1,0,0,0,0]])
W_in = np.random.randn(7,3)
W_out = np.random.randn(3,7)
in_layer0 = MatMul(W_in)
in_layer1 = MatMul(W_in)
out_layer = MatMul(W_out)

h0 = in_layer0.forward(c0)
h1 = in_layer1.forward(c1)
h = 0.5*(h1+h1)
s = out_layer.forward(h)

print(s)
# %%
text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
print(corpus)
print(id_to_word)
# %%
def create_contexts_target(corpus, window_size=1):
    target = corpus[window_size:-window_size]
    contexts = []

    for idx in range(window_size, len(corpus)-window_size):
        cs = []
        for t in range(-window_size,window_size+1):
            if t==0:
                continue
            cs.append(corpus[idx+t])
        contexts.append(cs)
    return np.array(contexts), np.array(target)
contexts, target = create_contexts_target(corpus,window_size=1)
# %%
