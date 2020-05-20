# Feature Transformation Ensemble Model with Batch Spectral Regularization for Cross-Domain Few-Shot Classification

Code release for Cross-Domain Few-Shot Learning (CD-FSL) Challenge.

## Enviroment

Python 2.7.16

Pytorch 1.0.0

## Steps

1. Prepare the source dataset miniImageNet and four target datasets CropDisease, EuroSAT, ISIC and ChestX.

2. Modify the paths of the datasets in `configs.py` according to the real paths.

3. Train base models on miniImageNet (pre-train)

    • *Train single model*
    
    ```bash
    python ./train_bsr.py --model ResNet10 --train_aug
    ```
    
    • *Generate projection matrices and train ensemble model*
    
    ```bash
    python ./create_Pmatrix.py
    python ./train_Pbsr.py --model ResNet10 --train_aug
    ```
4. Fine-tune and test for the 5-shot task in CropDisease as an example (change `n_shot` parameter to 20, 50 for 20-shot and 50-shot evaluations and change `dtarget` parameter to EuroSAT, ISIC, ChestX for the other target domains)

    • *Test the BSR and BSR+LP methods for single model*
    
    ```bash
     python ./finetune_lp.py --model ResNet10 --train_aug --use_saved --dtarget CropDisease --n_shot 5
    ```
    
    The `use_saved` flag is used to test with our saved models. You can close it to test with the reproduced models.
    
    Example output:
    
        BSR: 600 Test Acc = 92.17% +- 0.45%
        BSR+LP: 600 Test Acc = 94.45% +- 0.40%
    
    • *Test the BSR+DA and BSR+LP+DA methods for single model*
    
    ```bash
     python ./finetune_lp_da.py --model ResNet10 --train_aug --use_saved --dtarget CropDisease --n_shot 5
    ```
    
    Example output:
    
        BSR+DA: 600 Test Acc = 93.99% +- 0.39%
        BSR+LP+DA: 600 Test Acc = 95.97% +- 0.33%
    
    • *Test the BSR+ENT (not reported in the manuscript) and BSR+LP+ENT methods for single model*
    
    ```bash
     python ./finetune_lp_ent.py --model ResNet10 --train_aug --use_saved --dtarget CropDisease --n_shot 5
    ```
    
    Example output:
    
        BSR+ENT: 600 Test Acc = 94.24% +- 0.39%
        BSR+LP+ENT: 600 Test Acc = 95.69% +- 0.35%
    
    • *Test the BSR and BSR+LP methods for ensemble model*
    
    ```bash
     python ./finetune_P_lp.py --model ResNet10 --train_aug --use_saved --dtarget CropDisease --n_shot 5
    ```
    
    Example output:
    
        BSR (Ensemble): 600 Test Acc = 93.54% +- 0.41%
        BSR+LP (Ensemble): 600 Test Acc = 95.48% +- 0.38%
    
    • *Test the BSR+DA and BSR+LP+DA methods for ensemble model*
    
    ```bash
     python ./finetune_P_lp_da.py --model ResNet10 --train_aug --use_saved --dtarget CropDisease --n_shot 5
    ```
    
    Example output:
    
        BSR+DA (Ensemble): 600 Test Acc = 94.80% +- 0.36%
        BSR+LP+DA (Ensemble): 600 Test Acc = 96.59% +- 0.31%
    
    • *Test the BSR+ENT (not reported in the manuscript) and BSR+LP+ENT methods for ensemble model*
    
    ```bash
     python ./finetune_P_lp_ent.py --model ResNet10 --train_aug --use_saved --dtarget CropDisease --n_shot 5
    ```
    
    Example output:
    
        BSR+ENT (Ensemble): 600 Test Acc = 94.57% +- 0.40%
        BSR+LP+ENT (Ensemble): 600 Test Acc = 96.04% +- 0.36%
