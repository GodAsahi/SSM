# SSM
Scaffold-based deep generative model considering molecular stereochemical information

Required libraries：
Python 3.8
PyTorch 1.9
RDKit 2021
TensorFlow 2.7
NumPy 1.21

Before running, you need to prepare the training set molecules and the physical and chemical properties corresponding to the molecules of the training set.
When running, please do the following sequence：
extract_scaffold_from_dataset.py --> train.py --> sample.py
In which you need to provide the physical and chemical properties corresponding to each scaffold, and combine the extracted scaffolds with the training set molecules to form a dataset,and provide parameters inside each file as required.

The model is being updated to commercial software, please wait...
