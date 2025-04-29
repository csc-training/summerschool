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

- **loss function** measures how good/bad a model’s predictions are compared to the actual results: $\mathrm{E}=\frac{1}{2}\sum_{\mathrm{j}}\left[\mathrm{T}_{\mathrm{j}}-\mathfrak{\varphi}_{\mathrm{oj}} \right]^2$
- choose $\mathrm{w}_{ \mathrm{i,j,l}}$ (weight $i$ in neuron $j$ in layer $l$) that minimize the **loss function**, i.e.
   $\frac{\partial \mathrm{E}} {\partial \mathrm{w}_{\mathrm{i,j,l}}}=0$
- **training** is an interative **gradient descent** process: $\frac{\partial \mathrm{w}_{\mathrm{i,j,l}}}{\partial t}=- \frac{\partial \mathrm{E}}{\partial \mathrm{w}_{\mathrm{i,j,l}}}$
  -  training is done using labeled/known data (&#x1F91E; the model works for new data)
  
**Not guaranteed to find the true minima!**

# Derivative for one Layer with One Neuron

- derivative of the loss function depends on weigths, input, and true value
- forward pass is $\mathfrak{\varphi}_{\mathrm{o1}}=f_{\mathrm{1}}(w01+\mathfrak{\Sigma}_{\mathrm{i}}{\mathrm{x}_i\mathrm{w}_{\mathrm{i,1}}})$
- Apply the chain rule:
     - $\frac{\partial \mathrm{E}} {\partial \mathrm{w}_{\mathrm{i,j}}} = $\frac{\partial \mathrm{E}} {\partial \mathfrak{\varphi}_{\mathrm{oj}}} $
     - $\frac{\partial \mathrm{E}} {\partial \mathrm{w}_{\mathrm{i,j}}} = $\frac{\partial \mathrm{E}} {\partial \mathfrak{\varphi}_{\mathrm{oj}}} \times $\frac{\partial \mathfrak{\varphi}_{\mathrm{oj}}} {\partial \mathrm{w}_{\mathrm{i,j}}}$





# Data


# Summary
