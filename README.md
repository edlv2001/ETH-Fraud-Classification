# ETH-Fraud-Classification
 Compilation of state of art, and own proposed models for Ethereum EOA fraud classification.

## Data Retrieval
The Dataset used in this project consists on two different datasets merged:
1. [Ethereum Fraud Detection Dataset](https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset)
2. [Ethereum Fraud Dataset](https://www.kaggle.com/datasets/gescobero/ethereum-fraud-dataset)

The merged dataset can be found in this project, in the path: ./Data/Merged_Dataset.csv

### All Transactions Datasets:
All transactions were extracted from each dataset. These can be found here:
1. [Dataset 1 + Dataset 2 Part 1](https://drive.google.com/drive/folders/13AF1hXQvHs5OUWBWS3Geiv6612mzwTH4?usp=sharing)
2. [Dataset 2 Part 2](https://drive.google.com/drive/folders/1WbNFESZq6R8T60K9bpsYwWyiNftZ4JiF?usp=sharing)

From each of these, a set of time series was created. Transactions in Dataset 2 do not include any duplicated address from Dataset 1.

### Time Series Dataset
From each transaction list, a set of Time Series was created and serialized as Numpy arrays using Pickle. The Time Series sets associated with each dataset are:

- ./data/sequences.pkl
- ./data/sequences_2.pkl

Note: You may need Git LFS to download sequences_2.pkl.
