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

### TODO 

### Baseline Softmax Attention Transformer
* [ ] Write logs 
* [ ] Write preds to disk for each word in the vocabulary at each timestep 
(this will be needed later for comparison)
* [ ] Include callback to write metrics to Tensorboard?
* [ ] Write BLEU score at each timestep (free if logs written)
* [ ] Train tiny transformer with the above 
* [ ] Adapt tiny transformer with linear attention to have functionality 
which evaluates the KL-divergence between preds after training
    * For any word in the test input, 
    given the predicted probs P(x) and actual distribution Q(x)
    over the entire vocabulary, 
    we should calculate the discrete KL-divergence with -P(x) * log (Q(x)/P(x))
    * This means we need to store preds at every single output of the last normalized attention layer 
    at a cost of `O(nm)` 
    where `n = number of input tokens` and `m = target vocabulary size`
    
### Dataset stats 
Writing these down for my own personal sanity.

**2014 WMT English - German**
* newstest2014.en & newstest2014.de

| Attribute        | English           | German  |
| ------------- |:-------------:| -----:|
| sentences      | 3003 | 3003 |
| # words      | 59325      |  54865 |
| vocab size | 14098      |    16818 |
