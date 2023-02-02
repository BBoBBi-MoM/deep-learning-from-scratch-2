#%%
import sys
sys.path.append('..')
import numpy as np
from common.util import preprocess,create_contexts_target,convert_one_hot
from common.layers import MatMul, SoftmaxWithLoss
#%%
text = 'You say goodbye, and I say hello'
corpus, word_to_id, id_to_word = preprocess(text)
#%%
contexts, target = create_contexts_target(corpus, window_size=1)
#%%
vocab_size = len(word_to_id)
target = convert_one_hot(target,vocab_size)
contexts = convert_one_hot(contexts,vocab_size)
# %%
class SimpleCBOW:
    def __init__(self,vocab_size,hidden_size):
        V,H = vocab_size,hidden_size

        W_in = 0.01*np.random.randn(V,H).astype('f')
        W_out = 0.01*np.random.randn(H,V).astype('f')

        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()

        layers = [self.in_layer0,self.in_layer1, self.out_layer]
        self.params, self.grads = [],[]
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
        
        self.word_vecs = W_in

    def forward(self,contexts,target):
        h0 = self.in_layer0.forward(contexts[:,0])
        h1 = self.in_layer1.forward(contexts[:,1])

    def backward(self,dout=1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer1.backward(da)
        self.in_layer0.backward(da)
        return None
#%%
from common.trainer import Trainer
from common.optimizer import Adam
from common.util import preprocess, create_contexts_target,convert_one_hot

window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 1000

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

vocab_size = len(word_to_id)
contexts, target = create_contexts_target(corpus, window_size)
target = convert_one_hot(target,vocab_size)
contexts = convert_one_hot(contexts, vocab_size)

model = SimpleCBOW(vocab_size,hidden_size)
optimzier = Adam()
trainer = Trainer(model, optimzier)

trainer.fit(contexts,target,max_epoch,batch_size)
trainer.plot()
# %%
