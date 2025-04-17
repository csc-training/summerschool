---
title:  Introduction to Machine Learning
event:  CSC Summer School in High-Performance Computing 2025
lang:   en
---

# Machine Learning

- algorithms that learn patterns from data to make predictions on unseen data


  **The unreasonable effectiveness of data in machine learning!**

# Overfitting, Generalization, Regularization

- quality of a model is measured on new, unseen samples
- models with too many parameters can overfit to training data
- overfitting can be prevented

# Artificial Neurons

# Neural Networks

# Pytorch Example

<div class="column"  style="width:65%">
```python
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        return self.layers(x)
```
</div>

<div class="column"  style="width:33%">
 lol
</div>

Input of the model is of size 728,  the output is 10 (classes), and there are two layers in between. Final result is given by softmax operation. 
$y(\mathbf(z)=\frac{e^[{z_j}{\Sigma_k e^{z_k}}$

# Learning as an Optimization Problem

# Forward Pass

# Backward Pass


# Summary
