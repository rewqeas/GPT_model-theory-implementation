# GPT Training: Core Concepts

---

## 1. Why Positional Encoding is Added

Attention is **permutation invariant** by default — the order of tokens does not matter to the raw attention mechanism. For example:

$$\{1, 2, 3\} = \{3, 1, 2\} = \{2, 3, 1\}$$

In token terms:
> *"Dogs bites food"* = *"Food bites dog"*

Because vectors are contextualized and re-ordered meaning is lost, the model has no way to distinguish sequence order.

**Positional encoding** solves this by injecting ordering information into each token's representation, making the model **order-aware**.

---

## 2. Why Causal Masking Alone Is Not Enough

Causal masking restricts which tokens a position can attend to:

$$P(x_t \mid x_{\lt t}) = P(x_t \mid x_1, x_2, \dots, x_{t-1})$$

When $t = 28$ (context window boundary), the model attends to all previous tokens. However, **without positional encoding**, the model cannot distinguish between:

$$P(x_t \mid x_1, x_2, x_2, x_3, \dots, x_t) \quad \text{(randomized order)}$$

and

$$P(x_t \mid x_1, x_2, \dots, x_n) \quad \text{(correct serial order)}$$

Causal masking controls *which* tokens are visible; positional encoding tells the model *where* those tokens sit in the sequence. Both are necessary.

---

## 3. Training Methods

### 3.1 Teacher Forcing

During training, the model is given **ground truth tokens** as input at each step, regardless of what it previously predicted:

$$P(x_t \mid x_{\lt t})$$

If the model predicts the wrong token, the ground truth replaces it for the next step. This:
- Ensures stable, parallelizable training
- Prevents error compounding during the forward pass

### 3.2 Causal Masking

For a sequence of length $T$, all **future tokens are masked**, forcing the model to predict $x_t$ based only on $x_1, \dots, x_{t-1}$. This is what makes the task **autoregressive**.

---

## 4. The Unsupervised Pre-training Objective (Cross-Entropy Loss)

Although GPT pre-training is called *unsupervised*, each token acts as its own label. The loss function is:

$$L = -\sum_{i=1}^{M} \log P(x_i \mid x_{i-k}, \dots, x_{i-1})$$

or equivalently in GPT notation:

$$L = \sum_{i=1}^{m} \log P(u_i \mid u_{i-k}, \dots, u_{i-1};\, \theta), \quad u_k < u_i$$

For every position $u_i$ in the context window, the model predicts $u_i$ and computes loss against the actual token as label:

$$L = -\sum_{i=1}^{M} y_a \log(y_p)$$

where $y_a$ is the actual token (one-hot label) and $y_p$ is the predicted probability.

### Why Cross-Entropy Prevents the Model from "Forgetting"

- Each token in the sequence serves as a **supervised label**, so the loss is computed at every position.
- If the model miscalculates a token, **cross-entropy loss increases sharply** (due to $-\log p$ diverging as $p \to 0$), which destabilizes gradient descent.
- The optimizer is forced back toward the last state where loss was minimal — this acts as a memory-preserving constraint.

---

## 5. Residual Connections

### Formula (ResNet-style)

$$y = x + f(x)$$

where $x$ is the input (carrying the original representation) and $f(x)$ is the transformation applied by the current layer.

### What They Do

- The **original information is preserved** via the identity path $x$.
- Only the *difference* (residual) $f(x)$ needs to be learned by the layer.
- Weight and bias adjustments target the **error reduction**, not full relearning.

### Role of ReLU in $f(x)$

Inside $f(x)$, a **ReLU activation** is applied:

$$\text{ReLU}(z) = \max(0, z)$$

This means **only positive residuals** pass through $f(x)$ and contribute to the output. Negative (unhelpful) updates are suppressed, so only beneficial changes accumulate.

### Summary

Residual connections work together with the cross-entropy loss to ensure:
1. Gradients flow cleanly through deep networks (no vanishing gradients).
2. Correct information is **preserved** across layers.
3. Only updates that reduce loss propagate meaningfully.

---

## Quick Reference

| Concept | Core Idea |
|---|---|
| Teacher Forcing | Ground truth fed at each step → parallelizable, stable training |
| Causal Masking | Future tokens masked → autoregressive generation |
| Positional Encoding | Adds order info → breaks permutation invariance |
| Cross-Entropy Loss | Each token is its own label → large penalty for forgetting |
| Residual Connections | $y = x + f(x)$ → preserves information, eases gradient flow |
| ReLU in residuals | Only positive changes pass through $f(x)$ |
