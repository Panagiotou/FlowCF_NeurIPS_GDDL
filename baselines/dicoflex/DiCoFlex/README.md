# DiCoFlex

This repository is dedicated to the research and development of **DiCoFlex**.


## Abstract

Counterfactual explanations play a pivotal role in explainable artificial intelligence (XAI) by offering intuitive, human-understandable alternatives that elucidate machine learning model decisions. Despite their significance, existing methods for generating counterfactuals often require constant access to the predictive model, involve computationally intensive optimization for each instance and lack the flexibility to adapt to new user-defined constraints without retraining. In this paper, we propose DiCoFlex, a novel model-agnostic, conditional generative framework that produces multiple diverse counterfactuals in a single forward pass. Leveraging conditional normalizing flows trained solely on labeled data, DiCoFlex addresses key limitations by enabling real-time user-driven customization of constraints such as sparsity and actionability at inference time. Extensive experiments on standard benchmark datasets show that DiCoFlex outperforms existing methods in terms of validity, diversity, proximity, and constraint adherence, making it a practical and scalable solution for counterfactual generation in sensitive decision-making domains.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Code Structure](#code-structure)
- [Data](#data)
- [Experiments](#experiments)
- [Citation](#citation)
- [Contact](#contact)

## Introduction

Counterfactual explanations (CFs) have become an integral part of explainable artificial intelligence (XAI) by providing human-interpretable insights into complex machine learning models. CFs answer critical "what-if" questions by suggesting minimal and meaningful changes to input data that could alter the outcome of a predictive model. Such explanations have valuable applications across sensitive domains, including finance, healthcare, and legal decisions, where understanding the predictions of the model and potential alternative outcomes is paramount.

DiCoflex method:

1. A model-agnostic framework that generates multiple diverse counterfactuals in a single forward pass
2. Real-time customization of constraints including sparsity and actionability without retraining
3. Superior performance in terms of validity, proximity, diversity, and constraint compliance

Unlike existing methods that require complex optimizations or lack flexibility, DiCoFlex leverages conditional normalizing flows to learn the distribution of counterfactual explanations that inherently satisfy user-specified constraints, providing unprecedented flexibility while maintaining a balance between diversity and validity.

## Prerequisites

This section details the environment setup, including necessary libraries and frameworks. To clone the repository and set up the environment, use the following commands:

```shell
git clone git@github.com
cd counterfactuals
./setup_env.sh
```

## Code Structure

The repository is organized as follows to facilitate ease of use and contribution:

```
|── conf/                  # Configuration files
|── data/                  # Datasets
|── counterfactuals/       # Source code for the framework
|   ├── datasets/          # Counterfactual methods
|   ├── discriminative_models/  # Discriminative models for analysis
|   ├── generative_models/      # Generative models for analysis
|   ├── dicoflex/            # DiCoFlex method code
|   └── metrics/           # Evaluation metrics
|── README.md              # This document
└── ...
```

## Data

The full data folder can be found under the following link: [Link](data). More details regarding the datasets can be found in the paper in the appendix directory.

## Experiments

Execute the following scripts to train models and run experiments:

```shell
python3 counterfactuals/dicoflex/train_generic_counterfactual.py --all
```

## Citation
```
TBA
```
## Contact

In case of questions or comments please contact using LinkedIn: TBA
