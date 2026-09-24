# Where in the World? Country-Level Geolocation from Street-View Imagery (GeoGuessr Bot)

<!--[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<GITHUB_USER>/tcu-cosc-50523-final-project-geoguessrbot/blob/main/<NOTEBOOK_NAME>.ipynb) -->

**Team GoodWillTuning** · TCU COSC 50523 Deep Learning, Final Project

## Description

Give the model one street-level photo and it predicts which country the photo was taken in, like the game [GeoGuessr](https://www.geoguessr.com/). People who play GeoGuessr pick up on regional cues such as road markings, vegetation and architecture. Learning those cues is hard for a model, which is what makes this a good deep learning problem. Outside the game, the same technique is useful for intelligence gathering, journalism and organizing travel photos.

### Approach

| | |
|---|---|
| **Input** | A single RGB street-level photograph (640×640) |
| **Output** | Multiclass prediction over ~100 country classes |
| **Objective** | Minimize categorical cross-entropy loss |
| **Baseline** | Logistic regression classifier |
| **Main model** | Convolutional neural network, trained from scratch and/or fine-tuned from a pretrained backbone (e.g., ResNet) |
| **Data splits** | Train / validation / test split *within each country* so every class is represented in each split |
| **Controlled comparison** | Train and evaluate on progressively larger country subsets (10 → 50 → 100 classes) and/or compare training from scratch against transfer learning |

### Evaluation

- **Top-1 and Top-5 accuracy**: the standard metrics in prior geolocation work
- **Precision, recall and F1**, per class and aggregate
- **Confusion matrix** to find the country pairs the model mixes up most often
- **Target:** at best case scenario 70% Top-5 accuracy

### Dataset

[Streetview by Country](https://www.kaggle.com/datasets/sylshaw/streetview-by-country) (Kaggle): about 100,000 Google Street View images from roughly 100 countries (~1,000 per country), each with GPS coordinates.

## Demo

>  **Placeholder**

<!-- Example: [▶ Watch the demo](https://youtu.be/<VIDEO_ID>) -->

### Screenshots

>  **Placeholder**

<!--
![Sample prediction](docs/images/sample_prediction.png)
![Training curves](docs/images/training_curves.png)
![Confusion matrix](docs/images/confusion_matrix.png)
-->

## Getting Started

The project runs in **Google Colab**, so nothing needs to be installed locally.

### Prerequisites

- A Google account (for Colab and Google Drive)
- A [Kaggle](https://www.kaggle.com/) account and API token (`kaggle.json`)
- A GPU runtime is recommended (**Runtime → Change runtime type → GPU**)

### Setup

1. **Open the notebook in Colab.** Click the **Open in Colab** badge above, or go to **File → Open notebook → GitHub** and paste this repo's URL.
2. **Enable a GPU.** Go to **Runtime → Change runtime type → Hardware accelerator: GPU**.
3. **Add your Kaggle credentials.** Upload `kaggle.json` when the notebook asks for it, or save it in Colab **Secrets** as `KAGGLE_USERNAME` / `KAGGLE_KEY`.

4. **Run all cells.** Use **Runtime → Run all**.

### Project Structure

```
<!-- TODO: update as the project grows -->
├── data/
│   └── raw/              # Kaggle dataset goes here (gitignored; see scripts/download_data.py)
├── docs/                 # Proposal and reports
├── notebooks/            # Colab notebooks
├── scripts/
│   └── download_data.py  # Downloads the Kaggle dataset into data/raw/
└── README.md
```

## Authors

| Name | Email | Initial |
|---|---| --- |
| Tanner Temple | TANNER.TEMPLE@tcu.edu | TT |
| Esteban Hernandez-Anguiano | E.HERNANDEZ1798@tcu.edu | EH |
| Ethan Wong | E.W.WONG@tcu.edu | X |

Davis College of Science & Engineering, Texas Christian University, Fort Worth, TX

## Sources

1. T. Weyand, I. Kostrikov, and J. Philbin, "PlaNet - Photo Geolocation with Convolutional Neural Networks," in *Proc. European Conference on Computer Vision (ECCV)*, 2016.
2. J. Hays and A. A. Efros, "IM2GPS: Estimating Geographic Information from a Single Image," in *Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2008.
3. S. Shaw, "Streetview by Country," Kaggle. [Online]. Available: https://www.kaggle.com/datasets/sylshaw/streetview-by-country


