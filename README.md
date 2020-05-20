# Feature Transformation Ensemble Model with Batch Spectral Regularization for Cross-Domain Few-Shot Classification

Code release for Cross-Domain Few-Shot Learning (CD-FSL) Challenge.

## Enviroment

Python 2.7.16

Pytorch 1.0.0

## Steps

1. Train base models on miniImageNet (pre-train)

    • *Train single model*
    
    ```bash
    python ./train_bsr.py --model ResNet10 --train_aug
    ```
    
    • *Train ensemble model with our saved projection matrices*
    
    ```bash
    python ./train_Pbsr.py --model ResNet10 --train_aug --use_saved
    ```
    
    • *Train ensemble model with new projection matrices*
    
    ```bash
    python ./create_Pmatrix.py
    python ./train_Pbsr.py --model ResNet10 --train_aug
    ```
