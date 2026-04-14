# Data Foundations: Dimensions & Latent Space

---

## 1. Dataset Structure

In this implementation, training data is represented as a matrix $x$:

$$x \in \mathbb{R}^{n \times d}$$

| Symbol | Meaning |
|--------|---------|
| $n$ | Number of unique inputs (samples) |
| $d$ | Dimensionality of each input (feature space) |
| $x_i$ | Individual sample vector: $x_i \in \mathbb{R}^{1 \times d}$ |

---

## 2. Latent Vector Representation

When the input dimension $d$ is significantly larger than the target output ($d \gg \text{out}$), a **linear layer** projects the data into a **Latent Representation**.

This serves two critical purposes:

**Invariance**
It helps the model ignore "noise" (e.g., slight rotations or lighting changes) and focus on the identity of the object or token.

**Feature Modeling**
By distilling the data into critical metrics — the "essence" of the input — we drastically reduce computational overhead and prevent the model from memorizing redundant details.

---

## 3. The "Two Graphs" Perspective

To visualize optimization, we distinguish between the data we observe and the weights we tune:

### The Static Graph (Data Space)
- Defined by your **fixed input/output dimensions**
- Represents the physical "shape" of your data distribution
- Does **not** change during training

### The Weight Graph (Loss Landscape)
- An $n$-dimensional "terrain" where:
  - **Horizontal axes** = weights
  - **Vertical axis** = Loss
- **Gradient Descent** is performed here

---

## 4. Optimization Summary

Optimization is the bridge between the two graphs. Each gradient calculation and weight update follows this cycle:

```
1. Compute gradient
      → Find the steepest "downhill" direction on the Weight Graph

2. Update weights
      → Re-shapes the function in the Static Graph

3. Repeat over epochs
      → Model's prediction hyperplane converges to fit the data
```

Through iterative epochs, the model's prediction line (or hyperplane) converges until it fits the static data with **minimal error**.

---

> **Key Insight:** The Static Graph defines *what* the data looks like; the Weight Graph defines *how well* the model fits it. Gradient Descent navigates the Weight Graph to bring the model's predictions into alignment with the Static Graph.