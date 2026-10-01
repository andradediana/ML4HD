# Dilute Models, Not Results: Liquid State Machines for Shrinking Neural Network Parameter Counts

Hand gesture recognition from Google Soli radar data with Liquid State Machines (LSMs), Autoencoders, CNNs and RNNs.

**Authors:** Diego Alonso Brule Galleguillos and Diana Cristina Andrade Damian

Department of Mathematics, University of Padova

---

## Overview

Smaller neural networks matter for limited hardware, wearables and lower energy use. This project shows that a **Liquid State Machine** (a fixed, randomly initialised reservoir of spiking Leaky Integrate-and-Fire neurons) can replace large trainable components and drastically cut the number of **trainable parameters**, with only a small loss in accuracy.

The task is classifying **11 hand gestures** from sequences of Range-Doppler images (RDIs) recorded by Google's Soli radar. Our best trade-off model, **AE+LSM**, reaches **93.45% test accuracy with only 3,339 trainable parameters**. That is about 0.1% of the trainable parameters of our CNN+RNN baseline (3.5M) and about 3% of those of the LSM-only model (111k).

The full write-up is in [`Andrade_Brule.pdf`](Andrade_Brule.pdf).

### Approach in one line

Instead of feeding raw radar images into a reservoir, feed it a compact feature vector produced by a pre-trained encoder. The LSM then needs far fewer weights, and only a small readout layer is trained.

---

## Repository contents

| File | Description |
|---|---|
| `Andrade_Brule.ipynb` | Combined notebook with preprocessing, all model definitions, LSM parameter search, training, cross-session validation and testing |
| `Andrade_Brule.pdf` | Project paper |

## Dataset

The project uses the **Soli dataset** (Wang et al., UIST '16), available from the authors' repository:
**https://github.com/simonwsw/deep-soli**

The data is not included here because of its size. Download it following the instructions in that repository.

- 5,500 `.h5` files, each with 4 channels (`ch0`–`ch3`, one per radar antenna) and a `label`.
- Each channel is a sequence of frames; each frame has 1,024 values, reshaped to a **32×32** RDI. A file becomes a tensor of shape `(frames, 32, 32, 4)`.
- File names follow `gestureID_sessionID_instanceID.h5`, so gestures and sessions can be filtered by path.
- 2,750 files are multi-user recordings (11 subjects, 10 sessions). The other 2,750 are single-user recordings (sessions 0, 1, 4, 7, 13, 14) used for cross-session and personalised evaluation.

The 11 gestures are: Pinch Index, Pinch Pinky, Finger Slide, Finger Rub, Slow Swipe, Fast Swipe, Push, Pull, Palm Tilt, Circle and Palm Hold. Sequences with label 11 are skipped by the data loaders.

---

## Pipeline

```
.h5 files -> GMM background removal -> .npz cache -> truncate/pad to 40 frames
          -> (non-spiking) min-max normalisation -> CNN / Autoencoder -> LSTM -> FC(11)
          -> (spiking)     binary spike encoding -> [encoder] -> LSM reservoir -> FC(11)
```

1. **Background removal:** a Gaussian Mixture Model with 2 components separates hand from background. It is fitted on the first frame of each sequence and applied to the rest as a mask. Results are cached as compressed `.npz` files.
2. **Normalisation** (non-spiking models): min-max scaling to [0, 1].
3. **Truncation and padding:** every sequence is set to 40 frames (first / middle / random truncation, zero padding at the end).
4. **Spike generation** (LSM models): RDI values are binarised with a threshold (`P = 1 if f > θ else 0`) and flattened to `(40, 4096)`.
5. **Classification:** sequence-level, with a final fully connected 11-unit layer (or scikit-learn SVM, Random Forest and Logistic Regression readouts for the LSM-only variant).

## Models

| Model | Description |
|---|---|
| **CNN+RNN** | 3 conv + max-pool blocks (32/64/128 filters) → 2 FC layers → LSTM → FC(11). Baseline inspired by Wang et al. (2016) |
| **AE+RNN** | Convolutional autoencoder trained separately; the frozen encoder replaces the CNN, followed by a smaller LSTM (256) |
| **LSM + readout** | Reservoir of 256 LIF units with an SVM (C=128) readout, no trainable layers |
| **LSM** | Reservoir followed by global average pooling and dense layers (BatchNorm + ReLU) |
| **CNN+LSM** | CNN feature extractor feeding the reservoir |
| **AE+LSM** | Frozen encoder feeding the reservoir, global average pooling, BatchNorm and an 11-unit output layer |

**Reservoir hyperparameters** (found with a per-dimension search over units, leak rate, inhibitory ratio, sparsity and input connectivity):

| Setting | Units | Leak rate | Inhibitory ratio | Sparsity | C_inp |
|---|---|---|---|---|---|
| Reference work (Tsang et al., 2021) | 490 | 0.2 | 0.2 | 0.3 | 0.05 |
| This project | 256 | 0.15 | 0.35 | 0.05 | 1.0 |

**Training setup:** Adam, sparse categorical cross-entropy, up to 50 epochs, early stopping with best-weight restoration, a gradient-norm monitor callback, and polynomial learning-rate decay for the RNN models. Seed is fixed at 69.

**Data splits:**
- *Standard:* random 50/25/25 split of the files (2,750 train / 1,375 validation / 1,375 test).
- *Leave-one-session-out cross-validation:* single-user sessions 0, 1, 4, 7 and 13 (sessions 13 and 14 merged) are used for training and validation with 5 folds of 10 epochs each. All other files (2,750) form the test set.

---

## Results

### Architecture comparison

| Model | Test accuracy | Trainable parameters |
|---|---|---|
| CNN+RNN | 0.9804 | 3,513,643 |
| AE+RNN | 0.9716 | 628,491 |
| LSM + readout (SVM) | 0.9654 | 0 |
| LSM | 0.9499 | 111,243 |
| CNN+LSM | 0.9121 | 97,323 |
| **AE+LSM** | **0.9345** | **3,339** |

### Personalised (cross-session) classifiers

| Model | Accuracy |
|---|---|
| AE+RNN | 0.8339 |
| AE+LSM | 0.7168 |

### Main findings

- **AE+LSM gives the best complexity-performance trade-off:** a 1.5-point accuracy drop compared with the LSM-only model, for a large reduction in trainable parameters.
- **Pooling matters for sequence-level prediction:** removing pooling layers dropped accuracy to 0.18, while average pooling reached 0.9846 in the CNN+RNN experiments.
- **Global average pooling after the LSM** gave a large accuracy jump for the LSM-only model.
- **Adding dense layers** between components did not help, so the best models are minimal.
- **Personalised classification** loses accuracy with fewer training files, and AE+RNN is more robust than AE+LSM there. More augmentation and longer training could help.

### General recommendation

Find the best feature extractor for your problem, feed a compact, informative vector into the LSM, and run a thorough parameter search for the reservoir, since a random reservoir's best settings are problem-specific.

---

## How to run

The notebook was developed on **Kaggle**, and its paths point to Kaggle inputs.

1. Download the Soli data from [deep-soli](https://github.com/simonwsw/deep-soli). You need the raw `.h5` files.
2. Update the paths in the notebook to where your files live:
   - `/kaggle/input/solidata/dsp` → your raw `.h5` folder
   - `/kaggle/input/train-preprocessed/data_preprocessed` → your `.npz` folder
3. **Generate the preprocessed cache once.** The `precompute_dataset(...)` call is commented out in the notebook. Uncomment it and run it to create the `.npz` files (this is slow).
4. Run the notebook top to bottom. Most training and parameter-search cells are commented out, because the trained models were saved and reloaded for testing. Uncomment the ones you want to reproduce. The autoencoder and its encoder are loaded from a saved `.keras` file, so retrain it first (the training cell is also commented out) or point the path to your own weights.
5. The testing section downloads pretrained models with `kagglehub` (`xxdiegoalonsoxx/model-1`). Replace these with your own saved models if you retrain.

A GPU is strongly recommended.

### Requirements

- Python 3
- `tensorflow` (Keras 3 style `.keras` models are used)
- `numpy`, `pandas`, `scipy`, `scikit-learn`
- `h5py`, `matplotlib`, `seaborn`, `tqdm`, `kagglehub`

```bash
pip install tensorflow numpy pandas scipy scikit-learn h5py matplotlib seaborn tqdm kagglehub
```

## Notes and limitations

- The notebook is a combined version of all experiments, and not every model is re-tested inside it.
- The GMM background-removal class is adapted from [akash18tripathi/Gaussian-Mixture-Models-for-Background-Extraction](https://github.com/akash18tripathi/Gaussian-Mixture-Models-for-Background-Extraction).
- LSMs are random and sensitive to their parameters, so results depend on the reservoir settings.
- Results come from a single run with a fixed seed.

## References

1. Tsang, I. J., Corradi, F., Sifalakis, M., Van Leekwijck, W., & Latré, S. (2021). Radar-Based Hand Gesture Recognition Using Spiking Neural Networks. *Electronics*, 10, 851.
2. Maass, W., Natschläger, T., & Markram, H. (2002). Real-Time Computing Without Stable States. *Neural Computation*, 14, 2531–2560.
3. Wang, S., Song, J., Lien, J., Poupyrev, I., & Hilliges, O. (2016). Interacting with Soli: Exploring Fine-Grained Dynamic Gesture Recognition in the Radio-Frequency Spectrum. *UIST '16*.
4. Maass, W., & Markram, H. (2004). On the computational power of circuits of spiking neurons. *Journal of Computer and System Sciences*, 69, 593–616.
5. Hristov, B., et al. (2026). Leveraging convolutional sparse autoencoders for robust movement classification from low-density sEMG.
6. Tripathi, A. Gaussian mixture models for background extraction. GitHub.
