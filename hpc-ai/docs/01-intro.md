---
title:  Introduction to AI
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


#  Neural Networks

<div class="column"  style="width:58%">
  ![](img/NNetwork.png){width=90%}
</div>

<div class="column"  style="width:40%">

  - <small>the inputs and outputs are vectors </small>
  - <small>each layer $j$ is a matrix of size $l_{i}\times l_j$</small>
  - <small>RELU and softmax are functions operating on each value of the argument</small>

</div>

  - a prediction is comprised of a sequence of vector-matrix multiplications, each followed by an activation function call

    
# Forward Pass. Pytorch Example
 
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

Final result is given by softmax operation: $y_l(\mathbf{z})=\frac{e^{z_l}}{\Sigma_k e^{z_k}}$

# Learning as an Optimization Problem

- **loss function** measures how good/bad a model’s predictions are compared to the actual results: $\mathrm{E}_\mathrm{t}=\frac{1}{2}\sum_j\left[\mathrm{T}_{\mathrm{j}}-\mathfrak{\varphi}_{\mathrm{oj}} \right]^2$
- choose $\mathrm{w}_{\mathrm{i,j}}$ that minimize the **loss function**
   $\frac{\partial \mathrm{E}_{t}}{\partial f f\mathrm{w}_{\mathrm{i,j}}}=0$
- training 

# Backward Pass for One Neuron



# Data


# Summary
