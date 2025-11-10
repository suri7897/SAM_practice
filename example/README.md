# Cifar10 with WRN / CNN 🌁

This folder contains simple implementations of **Wide-ResNet** and **Simple CNN** models that can be trained on the **CIFAR-10** dataset using **Sharpness-Aware Minimization (SAM)**. 

## Model Structure

### Simple CNN
<img width="500" height="400" alt="simple_cnn" src="https://github.com/user-attachments/assets/682b0900-3744-4e99-ae72-7e51efcdcf38" />

**Model Description:**
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

<img width="590" height="390" alt="valid_loss" src="https://github.com/user-attachments/assets/127e593b-9ff2-493c-a6d3-6150d669cd4d" />
<img width="590" height="390" alt="valid_acc" src="https://github.com/user-attachments/assets/504808b5-07d2-4019-935c-9b94316bb2e8" />

