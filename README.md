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
2. Fine-tune and test for the 5-shot task in CropDisease as an example (change 'n_shot' parameter to 20, 50 for 20-shot and 50-shot evaluations and change 'dtarget' parameter to EuroSAT, ISIC, ChestX for the other target domains)

    • *Test the BSR and BSR+LP methods*
    
    ```bash
     python finetune_lp.py --model ResNet10 --train_aug --use_saved --dtarget EuroSAT --n_shot 5
    ```
    
    The ‘use_saved' flag is used to test with our saved models. You can close it to test with the reproduced models
    
    Example output:
    
        BSR: 600 Test Acc = 92.17% +- 0.45%
        BSR+LP: 600 Test Acc = 94.45% +- 0.40%
