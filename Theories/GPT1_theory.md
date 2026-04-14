# GPT-1
 
## Introduction
 
"Attention is all you need" paper started the revolution in the NLP field of AI. The paper introduced the **Transformer architecture**, which has an encoder-decoder block. It was invented for the sole purpose of language translation. However, for other NLP tasks, it required changes in the architectural design of the transformer. To solve the problem of re-designing blocks and training models from scratch, the **GPT-1 paper** was launched.
 
This paper introduced **transfer learning / fine-tuning**, which refers to using an existing trained model with additional data for a task-specific purpose, with little to no change in the model architecture.
 
- Built upon the **decoder block** of the transformer model
- Learns generalizable representations from unlabeled data via **unsupervised pre-training**
 
---
 
## Further Introduction
 
### Unsupervised Learning Approach
 
The model is initially fed a large corpus of **unlabeled data** — an unsupervised learning approach. The key change introduced was an **auxiliary task**: predicting the next word.
 
**Why?**
> This approach acted as regularization and helped the deep learning network generalize data more clearly.
 
### Other Approaches (at the time)
 
- People fed large corpora of text into **LSTM models** with many NLP auxiliary tasks (NER, few-shot learning, etc.). Despite outperforming state-of-the-art models at that time, LSTMs failed to handle **long-term dependencies** — hence transformers are better.
- Others tried adding many auxiliary tasks to models, achieving better performance but at **significantly higher computational cost**.
 
### Conclusion
 
> A Transformer model relying solely on its **unsupervised pre-training** with just **one auxiliary task** performed better and was substantially cheaper.
 
---
 
## Mathematics / Framework
 
The process consists of two parts:
 
---
 
### Part A — Unsupervised Pre-training (Auxiliary Task: Predict Next Word)
 
For an unsupervised corpus of tokens $U = \{u_1, u_2, \ldots, u_n\}$, the auxiliary ML task is predicting the next word. This is a **language modeling framework** known as an **Auto-regressive Model**.
 
$$L_1(U) = -\sum_{i} \log P(u_i \mid u_{i-k}, \ldots, u_{i-1};\, \Theta) \tag{a}$$
 
Where:
- $\Theta$ = all training parameters used in the network
- $k$ = context window size
 
This is the **cross-entropy loss** for unsupervised learning. The probability of a token is computed per unit within the context window.
 
---
 
#### Steps for Unsupervised Learning
 
**1. Token Embedding**
 
The token weight matrix $W_e$ has shape:
 
$$W_e \in \mathbb{R}^{v \times d}$$
 
Where:
- $v$ = total vocabulary size
- $d$ = embedding dimensions per token
 
For a token $u_i$ (a row index):
 
$$x_i^{(\text{tok})} = W_e[u_i] \in \mathbb{R}^{1 \times d} \tag{i}, \quad u_i \in U$$
 
**2. Positional Embedding**
 
The positional weight matrix $W_p$ has shape:
 
$$W_p \in \mathbb{R}^{T \times d}$$
 
Where:
- $T$ = total sequence length of the corpus
- $d$ = embedding dimensions
 
Position of the $i$-th word:
 
$$p_i \in \mathbb{R}^{1 \times d} \tag{ii}$$
 
**Final token representation** (token + position):
 
$$x_i = x_i^{(\text{tok})} + p_i$$
 
> **Note:** Sinusoidal (sine/cosine) functions are used for positional encoding because they are **bounded functions** in the range $(-1, 1)$.
 
---
 
**3. Transformer Blocks**
 
For every query, key, and value vector:
 
$$h_0 = U W_e + W_p$$
 
$$h_l = \text{transformer\_block}(h_{l-1}), \quad \forall\, l \in [1, n] \tag{b}$$ 

(applied for every layer $l$)
 
**Output projection** (logit probability scores for the $u$-th token):
 
$$P(u) = \text{softmax}(h_n W_e^T)$$
 
Expanding the softmax:
 
$$P(u \mid \text{context}) = \frac{e^{h_n \cdot e_i}}{\sum_j e^{h_n \cdot e_j}}$$
 
---
 
### Part B — Supervised Fine-tuning
 
Using the model from Part A, the model outputs the next token given a $k$-length context window (where $k < i$). For a curated dataset with tokens $x_1, \ldots, x_m$ and output label $y$, the final transformer block's output $h_l^m$ is compared against the actual output label.
 
$$P(y \mid x_1, \ldots, x_m) = \text{softmax}(h_L^m\, W_y) \tag{c}$$
 
Where:
- $m$ = total context window / total subwords in the model (task-specific token position)
- $h_L^m$ = output of the **last** transformer layer ($L$) at token position $m$
- $W_y$ = learned projection matrix mapping hidden states to label logits (trainable parameter, not from the dataset)
 
If there are 3 labels, the model calculates the probability score for which of the 3 labels should be chosen for the given context.
 
---
 
#### Supervised Loss
 
The loss function used is **cross-entropy**:
 
$$L_2(C) = \sum_{(x,\, y)} \log P(y \mid x_1, \ldots, x_m) \tag{d}$$
 
> Note: Since $y_i^{(\text{actual})} = 1$, it does not appear explicitly in the formula.
 
---
 
#### Why the Combined Framework Helps
 
The combined unsupervised + supervised framework — especially the language modeling (next token prediction) component — helped the model in two ways:
 
1. **Improved generalization** of next-token prediction (higher probability for the correct token)
2. **Accelerated / improved convergence** to the global minima
 
---
 
### Combined Loss Function
 
GPT-1 training is a **two-phase process** — $L_3$ is only active in Phase 2:
 
| | Phase 1 — Pre-training | Phase 2 — Fine-tuning |
|---|---|---|
| **Data** | Unlabeled corpus | Labeled task data |
| **Loss** | $L_1$ only | $L_3 = L_2 + \lambda L_1$ |
| **Backward pass on** | $L_1$ | $L_3$ (single joint pass) |
| **Purpose** | Learn language representations | Adapt to downstream task |
 
The combined loss used during fine-tuning:
 
$$L_3(C) = L_2(C) + \lambda \times L_1(C)$$
 
Where:
- $L_2(C)$ → supervised fine-tuning loss
- $L_1(C)$ → unsupervised LM loss computed **on the fine-tuning data** (acts as regularization to prevent catastrophic forgetting of pre-trained representations)
- $\lambda$ → weight coefficient *(hyperparameter)*
 
> **Important:** $L_3$ is a single combined backward pass during fine-tuning — it is **not** a re-run of Phase 1. The $L_1$ term here keeps the model grounded in its general language understanding while it adapts to the task.


 
