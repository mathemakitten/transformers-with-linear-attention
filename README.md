# transformers-with-linear-attention
https://openai.com/blog/requests-for-research-2/

#### Problem Statement
The Transformer model uses soft attention with softmax. If we could instead use linear attention (which can be converted into an RNN that uses fast weights), we could use the resulting model for RL. Specifically, an RL rollout with a transformer over a huge context would be impractical, but running an RNN with fast weights would be very feasible. Your goal: take any language modeling task; train a transformer; then find a way to get the same bits per character/word using a linear-attention transformer with different hyperparameters, without increasing the total number of parameters by much. Only one caveat: this may turn out to be impossible. But one potentially helpful hint: it is likely that transformers with linear attention require much higher dimensional key/value vectors compared to attention that uses the softmax, which can be done without significantly increasing the number of parameters.

#### Notes
from [here](https://arxiv.org/abs/1609.05866)

* Softmax attention is prohibitive in large-scale applications which:
	* have long sequences (n >> k), 
	* have an extremely high amount m of queries (possibly necessary to be processed in real-time, as with reinforcement learning applications)
	* have strong memory constraints

Softmax attention currently scales on the order of O(n), where `n` refers to the sequence length.

**Desired attributes in linear attention**
* At test time, a fixed-size representation of the document (softmax attention uses O(n) memory).
* At training time, if there are m queries per document, an algorithm which does not scale in O(nm) but only in O(n)

#### Outstanding questions to be answered
* What in the world are RNNs with fast weights? (TODO: read this paper)
