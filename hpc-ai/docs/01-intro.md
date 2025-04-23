---
title:  Introduction to Machine Learning
event:  CSC Summer School in High-Performance Computing 2025
lang:   en
---


# Machine Learning

- learn patterns from data to make predictions on unseen data

<div class="column"  style="width:99%; text-align: center;">
  ![](img/ML_types.png){width=61%}
  
  **The unreasonable effectiveness of data in machine learning!**
</div>



# Generalization, Overfitting, Regularization

- quality of a model is measured on new, unseen samples
- too simple models fail to describe the model
- models with too many parameters can overfit to training data
- overfitting can be prevented by regularization

<div class="column"  style="width:99%; text-align: center;">
  ![](img/Under_Over_fitting.png){width=64%}

  <small>By Chabacano [[GFDL](https://www.gnu.org/licenses/fdl-1.3.html) or [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)], from Wikimedia Commons</small>
</div>

# Artificial Neurons

# Neural Networks

# Pytorch Example
Example of a model with 3 layers. 
<div class="column"  style="width:75%">
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

<div class="column"  style="width:23%; text-align: center;">
![](img/forward_pass.png){width=39%}
</div>

Final result is given by softmax operation: $y(\mathbf{z})=\frac{e^{z_j}}{\Sigma_k e^{z_k}}$

# Learning as an Optimization Problem

# Forward Pass

# Backward Pass


# Summary
