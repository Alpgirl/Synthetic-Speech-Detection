# Synthetic-Speech-Detection
This project aims to develop and evaluate a robust fake speech detection system by leveraging advanced feature extraction techniques and neural network architectures

The structure of repository:
```bash
├── Model
├── src
│   ├── feature_extr.py
│   ├── load_asvspoof19.py
│   ├── load_asvspoof21.py
│   ├── metrics.py
│   ├── models.py
│   ├── models3.py (for last tries doing TE_Resnet)
└── Synthetic_Speech_Detection_resnet18.ipynb
└── ssd_TE_ResNet_models3.ipynb
```

Folder ```Model/``` contains experiments with ResNet18, ```src/``` contains all source .py files. The main file ```./Synthetic_Speech_Detection.ipynb``` contains the whole code pipeline for model training and evaluation.

Also added experiements with TE_ResNet in ```src/``` in ```models3.py/```
