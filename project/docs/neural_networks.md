# Phase: Neural networks

## Goal
Understand what happens inside a neural network: the forward pass, the loss, gradients, backpropagation and gradient descent. We first write one by hand in numpy, then rebuild it in PyTorch, so the real/bogus CNN in the next phase holds no mystery.

**Files built:**
- `notebooks/neural_networks.ipynb`. This phase is about understanding, so the from-scratch network lives in the notebook on purpose. The project's real network is `models/define_realbogus_cnn.py` (next phase).

---

## How I thought about it
1. **Use a problem where a straight line fails.** Two interleaved half-moons. Logistic regression (from *ML basics*) gets 0.88; a network should do clearly better, and it does (0.94).
2. **Write every step by hand once.** After coding backpropagation yourself, `loss.backward()` in PyTorch is just a convenience, not magic.
3. **Verify, don't trust.** Hand-written gradients are easy to get wrong, so I checked them numerically. They matched to 8 decimal places.
4. **Show overfitting live.** A big network on 40 points shows exactly why the CNN training loop uses early stopping.

---

## Concepts

### A neuron
```
output = activation(w1·x1 + w2·x2 + … + b)
```
That's logistic regression when the activation is a sigmoid. A network stacks many of these.

### Why stacking helps
With **ReLU** (`max(0, z)`) as the activation, each hidden neuron contributes a "bent line". Adding 16 bent lines together gives a curved boundary. That's why the network's boundary bends around the moons while logistic regression's is straight.

Without a non-linear activation, stacked layers collapse into one big linear function. Depth would add nothing.

### Forward pass (our network)
```
z1 = X·W1 + b1        (400 × 16)   weighted sums of the 2 inputs
h  = ReLU(z1)                       hidden layer
z2 = h·W2 + b2        (400 × 1)
p  = sigmoid(z2)                    probability of class 1
```

### Loss: binary cross-entropy
`L = −mean( y·log p + (1−y)·log(1−p) )`, the same loss as in *ML basics*.

### Backpropagation = the chain rule, from the output backwards
- **Output:** for sigmoid + cross-entropy, ∂L/∂z2 = **p − y**. The sigmoid's slope cancels the log's slope, leaving the simple "prediction minus truth".
- **Output weights:** ∂L/∂W2 = hᵀ·(p − y) / n.
- **Back into the hidden layer:** error_hidden = (p − y)·W2ᵀ × ReLU′(z1), where ReLU′ is 1 if z1 > 0, else 0.
- **Hidden weights:** ∂L/∂W1 = Xᵀ·error_hidden / n.

Each layer's gradient reuses the one after it, so one backward sweep gives every gradient. That efficiency is why deep learning is practical.

### Gradient checking
Nudge one weight by ε = 10⁻⁵ and measure `(L(w+ε) − L(w−ε)) / 2ε`. It must equal the backprop gradient. Ours:
```
W1[0,3]: backprop −0.02896520   numeric −0.02896520
```
If you remove the `× ReLU′` term, the check fails immediately (exercise 3).

### Gradient descent
`w ← w − learning_rate × gradient`, repeated. Too small a learning rate is slow; too large (try 5.0) overshoots and diverges.

**Adam** (used for the CNN) is gradient descent that adapts each weight's step size using its recent gradients. It's a good default.

### Initialisation
Weights start small and **random**. If they all started equal, every neuron would get the same gradient and stay identical forever.

### PyTorch does the same four steps
```python
optimizer.zero_grad()        # clear old gradients
loss = loss_fn(model(X), y)  # forward + loss
loss.backward()              # backprop (autograd computes every gradient)
optimizer.step()             # gradient descent
```
`BCEWithLogitsLoss` combines the sigmoid and cross-entropy in one numerically stable step, so the model outputs raw **logits**. `torch.no_grad()` turns gradient tracking off when we only want predictions. (I hit this bug myself: a prediction function defined inside `no_grad` but called outside it.)

### Overfitting and early stopping
A 256×256 network trained on 40 points:
- training loss falls towards 10⁻⁵ (it memorises every point, noise included),
- validation loss is lowest at **step 22**, then climbs.

**Early stopping** keeps the model from the best validation step. Other remedies: more data, augmentation, dropout, a smaller network.

---

## Steps, in order
1. Made 600 "moons" points; 400 for training, 200 for testing.
2. Baseline: logistic regression → 0.88 test accuracy, with a straight boundary.
3. numpy network: `initialise`, `forward`, `loss`, `backward`.
4. Gradient check on three parameters → matches.
5. 3,000 steps of gradient descent → 0.94, with a curved boundary.
6. The same network in PyTorch → 0.94.
7. Overfitting demo with training and validation curves.

## Pitfalls
- Forgetting `optimizer.zero_grad()`: gradients from earlier batches add up.
- Calling `.numpy()` on a tensor that tracks gradients: wrap the call in `torch.no_grad()` or use `.detach()`.
- Judging by training loss: only validation loss tells you about new data.
- A learning rate that's too large: the loss jumps around or blows up.

## Quiz
1. Why can't two linear layers without an activation learn the moons?
2. Derive why ∂L/∂z2 = p − y for sigmoid + cross-entropy (hint: σ′(z) = σ(z)(1 − σ(z))).
3. What does the gradient check compare, and why use ε on both sides?
4. Why must initial weights be random?
5. In the overfitting plot, what would happen to *test* accuracy if we kept the step-1500 model?
