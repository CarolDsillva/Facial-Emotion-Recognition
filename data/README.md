# Dataset: FER-2013

The dataset isn't included in this repo (about 300 MB). Download it from Kaggle:

- **CSV version**, used by `01_fer2013_cnn.ipynb`: [FER-2013 challenge data](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data). Place it at `data/fer2013.csv`.
- **Image-folder version**, used by `02_cnn_vs_resnet50v2_experiments.ipynb`: [FER-2013 images](https://www.kaggle.com/datasets/msambare/fer2013). Extract it so you have `data/train/<emotion>/` and `data/test/<emotion>/`.

| Label | Emotion | Images |
|---|---|---|
| 0 | Angry | 4,953 |
| 1 | Disgust | 547 |
| 2 | Fear | 5,121 |
| 3 | Happy | 8,989 |
| 4 | Sad | 6,077 |
| 5 | Surprise | 4,002 |
| 6 | Neutral | 6,198 |

35,887 grayscale 48×48 images in total.
