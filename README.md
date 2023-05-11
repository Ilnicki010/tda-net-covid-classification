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

I couldn't reproduce the original dataset because the first dataset is dynamic and changed over time and the second is a big set of cases so the authors picked a random sample from it.

### Dataset used for recreation

I decided to go with this dataset from Kaggle: https://www.kaggle.com/datasets/fusicfenta/chest-xray-for-covid19-detection

It's based on the same two data sources the original work is based on. It's **balanced** and contains **288 images for training** and **60 in a validation set**.

## TDA

TDA (Topological Data Analysis) - way for analyzing (usually high-dimensional) data using topological features.

## Proposed networks

The authors proposed 3 architectures of neural networks using TDA and 1 base CNN.

### Base CNN

original: </br>
<img width="250" alt="Screenshot 2023-05-04 at 21 58 30" src="https://github.com/Ilnicki010/tda-net-covid-classification/assets/18630618/5153d4ec-1c89-4bf0-b0c4-c630401d39f9">

implemented: </br>
![baseline_model](https://github.com/Ilnicki010/tda-net-covid-classification/assets/18630618/eac86a9e-115a-4d0f-87d0-14b7fc523ba5)


### $TDA-Net_{1}$

original: </br>
<img width="250" alt="Screenshot 2023-05-04 at 20 46 07" src="https://github.com/Ilnicki010/tda-net-covid-classification/assets/18630618/e2a10fdb-e317-4d9c-8a38-b8d013bf89ed">

implemented: </br>
![first_tda_net](https://github.com/Ilnicki010/tda-net-covid-classification/assets/18630618/4af63e72-273c-4467-baf0-d770e0783053)


### $TDA-Net_{1,2}$

original: </br>
<img width="250" alt="Screenshot 2023-05-05 at 23 26 13" src="https://github.com/Ilnicki010/tda-net-covid-classification/assets/18630618/f92d93f1-96f6-453a-9c42-e55359e2ee3f">

implemented: </br>
![second_tda_net_model](https://github.com/Ilnicki010/tda-net-covid-classification/assets/18630618/43b83bd4-ebd9-4a0b-bcec-2e31ab387edc)


### $TDA-Net_{1,2,3}$

original: </br>
<img width="250" alt="Screenshot 2023-05-11 at 17 22 22" src="https://github.com/Ilnicki010/tda-net-covid-classification/assets/18630618/5528045c-e5a1-45ee-9655-9cd45341ec06">

implemented: </br>
![third_tda_net_model](https://github.com/Ilnicki010/tda-net-covid-classification/assets/18630618/68e913d6-e0cf-4975-a5a7-fd424c9f5dbf)


## Results and conclusions

The end results in the original paper look like this:

|           | Base model | $TDA-Net_{1}$ | $TDA-Net_{1,2}$ | $TDA-Net_{1,2,3}$ |
|-----------|------------|---------------|-----------------|-------------------|
| Accuracy  | 0.87       | 0.89          | 0.92            | 0.93              |
| Precision | 0.84       | 0.84          | 0.95            | 0.88               |
| Recall    | 0.87       | 0.87          | 0.85            | 0.95              |
| f-1 score | 0.86       | 0.86          | 0.90            | 0.92              |
| TNR       | 0.89       | 0.88           | 0.97            | 0.91               |


However, in my implementation I got the following results:

|           | Base model | $TDA-Net_{1}$ | $TDA-Net_{1,2}$ | $TDA-Net_{1,2,3}$ |
|-----------|------------|---------------|-----------------|-------------------|
| Accuracy  | 0.97       | 0.85          | 0.90            | 0.97              |
| Precision | 0.97       | 0.82          | 0.88            | 1.0               |
| Recall    | 0.97       | 0.90          | 0.93            | 0.93              |
| f-1 score | 0.97       | 0.86          | 0.90            | 0.97              |
| TNR       | 0.97       | 0.8           | 0.87            | 1.0               |

## Setting up

1. ```git clone https://github.com/Ilnicki010/tda-net-covid-classification.git```
2. ```cd tda-net-covid-classification```
3. Create ```data``` folder with datasets from here: https://www.kaggle.com/datasets/fusicfenta/chest-xray-for-covid19-detection
4. ```pip install -r requirements.txt```
5. Open ```main.ipynb``` run and analyze all cells
