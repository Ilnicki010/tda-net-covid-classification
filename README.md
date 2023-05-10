I tried to recreate results from "[TDA-Net: Fusion of Persistent Homology and Deep Learning Features for COVID-19 Detection From Chest X-Ray Images](https://arxiv.org/abs/2101.08398)" paper by Mustafa Hajij, Ghada Zamzmi, and Fawwaz Batayneh
 published on 3 Aug 2021.

## Goal

The problem we try to solve is a supervised binary classification of chest X-ray photos. There are 2 separate classes:
- "Covid" - patient is affected by COVID-19
- "Normal" - patient is healthy

Model's input: black and white X-ray image of a chest
Model's output: one of two classes: "Covid" or "Normal"

## Dataset

I wasn't able to exactly recreate the dataset used in a paper. The proposed dataset was built from two publicly available databases:

1. positive cases were taken from: https://github.com/ieee8023/covid-chestxray-dataset
2. normal cases: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

I couldn't represent the original dataset because the first dataset is dynamic and changed over time and the second is a big set of cases so the authors picked a random sample from it.

**Dataset used for recreation**

I decided to go with this dataset from Kaggle: https://www.kaggle.com/datasets/fusicfenta/chest-xray-for-covid19-detection

It's based on the same two data sources the original work is based on. It's **balanced** and contains **288 images for training** and **60 in a validation set**.

## TDA

## Proposed networks

The authors proposed 3 architectures of neural networks using TDA and 1 base CNN.

### Base CNN

### $TDA-Net_{1}$

### $TDA-Net_{1,2}$

### $TDA-Net_{1,2,3}$

## Results and conclusion

## Setting up

1. ```git clone https://github.com/Ilnicki010/tda-net-covid-classification.git```
2. ```cd tda-net-covid-classification```
3. Create ```data``` folder with datasets from here: https://www.kaggle.com/datasets/fusicfenta/chest-xray-for-covid19-detection
4. ```pip install -r requirements.txt```
5. Open ```main.ipynb``` run and analyze all cells