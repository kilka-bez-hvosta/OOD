# Trustworthy AI: Out-of-Distribution Detection in CNNs

This repository contains the code and experimental results of a comparative study on the sensitivity of classification architectures to Out-of-Distribution (OOD) data. The study evaluates the impact of augmenting training data with synthetic OOD examples on the quality of OOD detection.

## Abstract

Convolutional neural networks often exhibit overconfidence when processing OOD data, posing significant risks in safety-critical applications. This study presents a comparative experimental analysis of classification architectures regarding their sensitivity to OOD objects and evaluates the impact of augmenting training data with synthetic OOD examples. We implement and compare eight model configurations based on a unified feature extractor, including Softmax, Bayesian, Gaussian Naive Bayes, and One-vs-All (Unary) classifiers, with and without an additional OOD class. Using CIFAR-10 as in-distribution data and SVHN as OOD data, we assess performance via ROC-AUC based on Maximum Softmax Probability (MSP) and entropy metrics. Results demonstrate that the Unary CNN+OOD model achieves the highest OOD detection quality (AUROC 0.8802), significantly outperforming baseline Softmax (0.8553) and probabilistic approaches, while maintaining comparable classification accuracy. Furthermore, we investigate the effects of fine-tuning and regularization, revealing that unary models are prone to overfitting, which can be mitigated through dropout and weight decay, yielding further improvements (AUROC up to 0.8998).

## Objectives

1. Compare the sensitivity of different classification architectures to OOD data.
2. Evaluate the impact of synthetic OOD examples on model confidence scores.
3. Identify the most effective configuration for OOD detection tasks.

## Metrics

- AUROC (Area Under ROC Curve) based on:
  - Maximum Softmax Probability (MSP)
  - Entropy of the probability distribution
- Accuracy, Precision, Recall, F1-score (on in-distribution data)
- Confusion Matrix
- Reject Curves (FRR/TRR)

## Experimental Design

### Datasets

| Dataset      | Role                      |
|--------------|---------------------------|
| CIFAR-10     | In-Distribution (ID)      |
| SVHN         | Out-of-Distribution (OOD) |
| Uniform Noise| Synthetic OOD (for training with OOD class) |

### Models

| Model                     | Classifier       | OOD Class |
|---------------------------|------------------|-----------|
| CNN Softmax               | Softmax          | No        |
| CNN + GaussianNB          | GaussianNB       | No        |
| CNN Bayesian              | Bayesian         | No        |
| CNN + OOD                 | Softmax          | Yes       |
| CNN + GaussianNB + OOD    | GaussianNB       | Yes       |
| CNN + Bayesian + OOD      | Bayesian         | Yes       |
| Unary CNN                 | One-vs-All       | No        |
| Unary CNN + OOD           | One-vs-All       | Yes       |

### Architecture

All models share a common feature extractor, FeatureCNN, consisting of three convolutional blocks with ReLU activation and MaxPooling layers.

### Training

- Optimizer: Adam (lr=0.001)
- Loss functions:
  - CrossEntropyLoss (for multi-class models)
  - BCEWithLogitsLoss (for unary models)
- Regularization: Dropout and Weight Decay to mitigate overfitting

## Results

### Classification Performance (ID - CIFAR-10)

| Model                     | Accuracy | Precision | Recall | F1-score |
|---------------------------|----------|-----------|--------|----------|
| CNN Softmax               | 0.73     | 0.74      | 0.73   | 0.73     |
| Unary CNN                 | 0.75     | 0.75      | 0.75   | 0.75     |
| CNN + GaussianNB          | 0.58     | 0.58      | 0.58   | 0.57     |
| CNN + Bayesian            | 0.69     | 0.70      | 0.69   | 0.68     |

Adding an OOD class does not degrade classification accuracy.

### OOD Detection Performance (SVHN)

| Model                     | AUROC (Entropy) | AUROC (MSP) |
|---------------------------|-----------------|-------------|
| CNN Softmax               | 0.8553          | 0.8324      |
| Unary CNN + OOD           | 0.8802          | 0.7416      |
| Unary CNN (with regularization) | 0.8998     | 0.7786      |

The Unary CNN + OOD model achieved the highest AUROC (0.8802) among all configurations.

### Overfitting and Regularization

During fine-tuning, unary models showed degradation in OOD detection quality. The introduction of Dropout and Weight Decay stabilized training and improved AUROC to 0.8998.

## Visualizations

- ROC curves for the best-performing models
- MSP and entropy distributions for ID and OOD data
- Reject curves (FRR/TRR) before and after regularization

## Dependencies

- PyTorch
- scikit-learn
- NumPy
- Matplotlib / Seaborn
- Google Colab (for reproducibility)

## License

This project is distributed under the Creative Commons license.

## References

- [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [SVHN dataset](http://ufldl.stanford.edu/housenumbers/)
- Full paper (PDF) | Will be soon

## Conclusion
The Unary classifier with an explicit OOD class and regularization demonstrates the best OOD detection performance without sacrificing classification accuracy. The method is simple to implement and does not require complex post-hoc activation modifications, making it suitable for resource-constrained environments.
The Unary classifier with an explicit OOD class and regularization demonstrates the best OOD detection performance without sacrificing classification accuracy. The method is simple to implement and does not require complex post-hoc activation modifications, making it suitable for resource-constrained environments.
