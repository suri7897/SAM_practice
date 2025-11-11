# Cifar10 with WRN / CNN 🌁

This folder contains simple implementations of **Wide-ResNet** and **Simple CNN** models that can be trained on the **CIFAR-10** dataset using **Sharpness-Aware Minimization (SAM)**. 

## Model Structure

### Simple CNN

**Model Description:**

<img width="779" height="528" alt="cnn_model_structure" src="https://github.com/user-attachments/assets/1e569326-9441-4f56-b088-21e63a788e6d" />

- **Input:** 32×32×3 RGB image  
- **Convolutional Layers:**
  - Two 3×3 Conv blocks (32, 64 filters)  
  - Each followed by ReLU, 2×2 MaxPooling, and Dropout (p=0.25)  
- **Fully Connected Layers:**
  - Flatten → FC(4096) → ReLU → Dropout(p=0.5)  
  - FC(128) → Classification (10 classes)
- **Activation:** ReLU  
- **Regularization:** Dropout
- **Default learning rate** : 1e-2

## Usage

```bash
  # For usage of CNN, change --model=wrn to --model=cnn
   python3 train.py --model=wrn # For SAM
   python3 train_multi.py --model=wrn # For Multi_SAM
   python3 train_sgd.py --model=wrn # For SGD
```
## Results

After training, log files (.csv) are saved in the `results` folder.
You can visualize and save the loss and accuracy curves for each optimization method using the Jupyter notebook: `results/plotter.ipynb`.
Figures are saved in `figures` folder.


<img width="590" height="390" alt="valid_loss_cnn" src="https://github.com/user-attachments/assets/127e593b-9ff2-493c-a6d3-6150d669cd4d" />
<img width="590" height="390" alt="valid_acc_cnn" src="https://github.com/user-attachments/assets/504808b5-07d2-4019-935c-9b94316bb2e8" />

**[Figure 1,2 | figure of Validation Loss, Accuracy of CNN model]** 

<br>

<img width="590" height="390" alt="valid_loss_wrn" src="https://github.com/user-attachments/assets/d91e6290-f275-43bc-a068-e1bed32e015a" />
<img width="590" height="390" alt="valid_acc_wrn" src="https://github.com/user-attachments/assets/a054c74f-8a1c-4335-9e08-c7ff17189c71" />

**[Figure 3,4 | figure of Validation Loss, Accuracy of WRN model]** 



