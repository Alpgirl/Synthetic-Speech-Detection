# Synthetic-Speech-Detection
<p float="center">
  <img src="https://github.com/Alpgirl/Synthetic-Speech-Detection/assets/68153923/f1b1d652-a9a4-4a64-9c60-71226f87d232" height="200" alt="Mask">
</p>

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Details](#dataset-details)
- [Team Members](#team-members)
- [Repository Structure](#repository-structure)


## Project Overview
Current work is devoted to the development of a comprehensive fake speech
detection system based on state-of-the-art techniques in feature extraction and neural network architectures. \
We address the key challenges:
- Growing generation of highly realistic fake speech
- Poor generalization of synthetic speech detection systems 

In this project, we introduce our own fake speech detection system developed based on the framework presented in "Fake Speech Detection Using Residual Network with Transformer Encoder" [Zhang et. al.](https://doi.org/10.1145/3437880.3460408)

## Dataset Details
Our target datasets are [ASVspoof19](https://doi.org/10.48550/arXiv.1911.01601) and [ASVspoof21](https://doi.org/10.48550/arXiv.2109.00537). \
We conduct training and evaluation on ASVspoof19 and only evaluation on ASVspoof21. See the details in the report.


## Team Members
| Name              | Role                    | Contact Information |
|-------------------|-------------------------|---------------------|
| Inna Larina       | Project Lead, Developer | [Email](inna.larina@skoltech.ru) |
| Hernán Nenjer     | Developer               | [Email](hernan.nenjer@skoltech.ru) |
| Maksim Komiakov   | Developer               | [Email]() |
| Folu Obidare      | Developer               | [Email]() |
| Ilona Basset      | Developer               | [Email]() |

## Repository Structure
The structure of repository:
```bash
├── weights
│   ├── weights_resnet18_CQCC.pt
│   ├── weights_resnet18_LPS.pt
│   ├── weights_resnet18_MFCC.pt
├── src
│   ├── feature_extr.py
│   ├── load_asvspoof19.py
│   ├── load_asvspoof21.py
│   ├── metrics.py
│   ├── models.py
├── README.md
├── Final Report.pdf
└── Synthetic_Speech_Detection.ipynb
```

Folder ```weights/``` contains weights of trained ResNet18 model on ASVspoof19 dataset, ```src/``` contains all source .py files. The main file ```./Synthetic_Speech_Detection.ipynb``` contains the whole code pipeline for model training and evaluation.

