~~Breaking the Blindfold: Deep Learning-based Blind Side-channel Analysis

This repository contains datasets and code used in the paper "Breaking the Blindfold: Deep Learning-based Blind 
Side-channel Analysis." It is published for availability verification and will be updated based on feedbacks from  
Artifact Evaluation Committee (AEC) for functionality and reproducibility assessments.

## Datasets (Datasets.zip)
- Chipwhisperer 
- Kyber:
  - Original dataset credits: "Side-Channel Power Trace Dataset for Kyber Pair-Pointwise Multiplication on Cortex-M4."
  - If used, please cite: @article{rezaeezade2025side,
  title={Side-Channel Power Trace Dataset for Kyber Pair-Pointwise Multiplication on Cortex-M4},
  author={Rezaeezade, Azade and Yap, Trevor and Jap, Dirmanto and Bhasin, Shivam and Picek, Stjepan},
  journal={Cryptology ePrint Archive},
  year={2025}
}

- Ascon:
  - Original dataset credits: "Lightweight but Not Easy: Side-channel Analysis of the Ascon Authenticated Cipher on a 32-bit Microcontroller."
  - If used, please cite: Ascon SCA software and hardware databases (0.1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.10229484

## Code (Code.zip)
The provided Python scripts DL_blind_attack_kyber_chipwhisperer.py (for kyber and Chipwisperer datasets) and DL_blind_attack_Ascon.py (for Ascon dataset) 
implement deep learning-based blind side-channel attack (DL-BSCA) as introduced in the paper with all mentioned labeling techniques (Slicing, VA, and MC labeling techniques).

The provided Python scripts classic_blind_attack_kyber_chipwhisperer.py (for kyber and Chipwisperer datasets) and Classic_blind_attack_Ascon.py (for Ascon dataset) 
implement classic (without deep neural networks usage) blind side-channel attack (BSCA) with Slicing VA, and MC labeling techniques.

Scripts at the root level can be executed directly, while supporting scripts in the src folder are utilized by these main scripts.

For more details on methods please refer to the paper.

## Results (Results.zip)
This folder includes the best MLP and CNN models we found for each labeling technique using random search in a pool of 100 models.
It also includes the theoretical joint distributions calculated in the code for each dataset and empirical labels generated using each technique. 

