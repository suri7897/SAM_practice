# Sharpness Awareness Minimization(SAM) / Practice

> [!IMPORTANT]
I acknowledge that all ideas and codes presented here are derived from the study titled '**[Sharpness-Aware Minimization for Efficiently Improving Generalization](https://arxiv.org/abs/2010.01412)**' (2020). 
---

This repository contains the implementation of **Sharpness-Aware Minimization for Efficiently Improving Generalization (2020)** by Pierre Foret et al. 
Also, this repository contains the pytorch-code implementation of SAM implementation by [David Samuel](https://github.com/davda54).

For further details on original paper and codes, refer to the repository available at

Paper : https://arxiv.org/abs/2010.01412 

Codes : https://github.com/davda54/sam.git

## Usage

It additionally includes MultiSAM, which pertubates multiple times. 

```python
from sam_multi_step import MultiSAM
...

model = YourModel()
base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
optimizer = MultiSAM(model.parameters(), base_optimizer, lr=0.1, momentum=0.9, n_step=2)
...

for input, output in data:

# Same as Original SAM

  # first forward-backward pass
  loss = loss_function(output, model(input))  # use this loss for any training statistics
  loss.backward()
  optimizer.first_step(zero_grad=True)
  
  # second forward-backward pass
  loss_function(output, model(input)).backward()  # make sure to do a full forward pass
  optimizer.second_step(zero_grad=True)
...
```

## **Experiment**

  In `example` folder, you can do simple implementations of Wide-ResNet and Simple CNN models that can be trained on the CIFAR-10 dataset using Sharpness-Aware Minimization (SAM).
  For more information, refer to `README.md` file in `example` folder.

## **Explanation**

For your information, a presentation file titled **`SAM_explain(kor).pdf`** is included in this repository.  
It provides an overview of **SAM** and this experiment, and is written in **Korean**.
  
