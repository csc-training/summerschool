---
title:  Introduction to Machine Learning
event:  CSC Summer School in High-Performance Computing 2025
lang:   en
---


# Machine Learning

- learn patterns from data to make predictions on unseen data

<div class="column"  style="width:99%; text-align: center;">
  ![](img/ML_types.png){width=61%}
  
  <small>**The unreasonable effectiveness of data in machine learning!**</small>
</div>



# Generalization, Overfitting, Regularization

- quality of a model is measured on new, unseen samples
- too simple models fail to describe the model
- models with too many parameters can overfit to training data
- overfitting can be prevented by regularization

<div class="column"  style="width:99%; text-align: center;">
  ![](img/Under_Over_fitting.png){width=64%}

  <small>From Wikimedia Commons, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)</small>
</div>

# Artificial Neurons

<div class="column"  style="width:57%">
  ![](img/Neuron.png){width=125%}

  <small>From Wikimedia Commons, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)</small>

</div>
<div class="column"  style="width:39%; text-align: center;">
  ![](img/activation_functions.png){width=64%}

  <small>From Wikimedia Commons, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)</small>
</div>


# Feed-Forward Neural Networks

<div class="column"  style="width:55%">
  ![](img/NNetwork.png){width=70%}
</div>

<div class="column"  style="width:40%">
  <small></small>
  - the inputs is  a vector of size $n$
  - the outputs is a vector of size $p$
  - layer 1 is a matrix of size $n\timesl_1$
  - layer 2 is a matrix of size $l_1\times l_2$
  - layer 3 is a matrix of size $l_2\times p$
  - RELU and softmax are function operating on each value of the argument
  - a prediction is comprised of a sequence of vector-matrix multiplications, each followed by an activation function applied to the result
</div>small
  
</div>

# Pytorch Example
Example of a model with 3 layers. 
<div class="column"  style="width:75%">
```python
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(5, 4),
            nn.ReLU(),
            nn.Linear(4, 5),
            nn.ReLU(),
            nn.Linear(5, 3),
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
