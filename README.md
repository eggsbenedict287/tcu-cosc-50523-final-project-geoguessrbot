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

The project runs in **Google Colab**, so nothing needs to be installed locally. Environment setup is handled by [`notebooks/init.ipynb`](notebooks/init.ipynb), which clones this repo into the Colab runtime, configures git, and loads the dataset (caching it in your Google Drive).

### Prerequisites

- A Google account (for Colab and Google Drive). The notebook mounts your Drive to cache the zipped dataset, so make sure you have enough free space for it.
- A [Kaggle](https://www.kaggle.com/) account and API token. Your username and key are in the `kaggle.json` file you get from **Kaggle → Settings → API → Create New Token**.
- A GitHub account and a [Personal Access Token](https://github.com/settings/tokens) with write access to this repo, so you can push commits from Colab.
- A GPU runtime is recommended (**Runtime → Change runtime type → GPU**)

### Setup

1. **Open the setup notebook in Colab.** Go to **File → Open notebook → GitHub**, paste this repo's URL, and open `notebooks/init.ipynb`.
2. **Enable a GPU.** Go to **Runtime → Change runtime type → Hardware accelerator: GPU**.
3. **Add your Colab Secrets.** Click the  **Secrets** icon in Colab's left sidebar, add each secret below with **+ Add new secret**, and turn on **Notebook access** for every one:

   | Secret | Value | Used for |
   |---|---|---|
   | `GITHUB_USERNAME` | Your GitHub username | `git config user.name` |
   | `GITHUB_EMAIL` | The email on your GitHub account | `git config user.email` |
   | `GITHUB_TOKEN` | Your GitHub Personal Access Token | Authenticating the `origin` remote so you can push |
   | `KAGGLE_USERNAME` | Your Kaggle username (`username` in `kaggle.json`) | Downloading the dataset |
   | `KAGGLE_KEY` | Your Kaggle API key (`key` in `kaggle.json`) | Downloading the dataset |

4. **Run the "Set up GitHub repository" cell.** It clones the repo into `/content/tcu-cosc-50523-final-project-geoguessrbot` (or runs `git pull` if it's already there) and sets up your git identity and credentials.
   > **Working on a branch other than `main`?** Check it out manually (`!git checkout <branch>`) before running the next cell.
5. **Run the dataset cell.** It mounts Google Drive (authorize access when prompted), then:
   - **First run:** downloads the dataset from Kaggle with `scripts/download_data.py` and saves a zip of it to `MyDrive/tcu-cosc-50523-final-project-geoguessrbot/streetview-by-country.zip`. This is slow, but you only have to do it once.
   - **Later runs:** copies the cached zip from Drive and unzips it into `data/raw/`.

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
| Ethan Wong | E.W.WONG@tcu.edu | EW |

Davis College of Science & Engineering, Texas Christian University, Fort Worth, TX

## Sources

1. T. Weyand, I. Kostrikov, and J. Philbin, "PlaNet - Photo Geolocation with Convolutional Neural Networks," in *Proc. European Conference on Computer Vision (ECCV)*, 2016.
2. J. Hays and A. A. Efros, "IM2GPS: Estimating Geographic Information from a Single Image," in *Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2008.
3. S. Shaw, "Streetview by Country," Kaggle. [Online]. Available: https://www.kaggle.com/datasets/sylshaw/streetview-by-country


