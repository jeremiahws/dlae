# Deep Learning Application Engine (DLAE)

DLAE is a software framework and application that enables users to design, train, validate, and coherently encapsulate and deploy deep learning (DL) models in medical imaging, while hiding programmatic implementation details. Researchers and developers in other imaging domains may also find this software useful.

# Python Version
DLAE currently uses `python==3.6.8`

# Installation

The following modules are required to run DLAE:
```markdown
tensorflow-gpu==1.13.1
keras==2.2.4
imageio==2.5.0
opencv==3.4.2
keras-applications==1.0.7
scikit-learn==0.21.2
pillow==6.1.0
```

To install using `anaconda3`
```markdown
conda create -n dlae python==3.6.8
conda install tensorflow-gpu==1.13.1 keras==2.2.4 imageio==2.5.0 opencv==3.4.2 keras-applications==1.0.7 scikit-learn==0.21.2 pillow==6.1.0
```

# Usage

There are two primary usages (1) GUI mode and (2) silent mode. For GUI mode, call DLAE with the following command
```markdown
python dlae.py
```
For silent mode, call dlae.py with a configuration file as the input
```markdown
python dlae.py config_file.json
```

# Test Model Compilation

To test the compilation of an example model, run DLAE in GUI mode and load one of the prebuilt models (either a CNN, FCN, GAN, or BBD). Once the layer list is populated, click `Test model compilation`.

# Example Datasets

Example datasets for each of the DL techniques currently included in DLAE are provided in `dlae/datasets`. These were created by simply making matrices or random noise that correspond to the required input dimensions. The primary purpose of these datasets is to familiarize users with the data input formats for the four different DL techniques. Users can use these datasets to guide the curation of their own data sets.
