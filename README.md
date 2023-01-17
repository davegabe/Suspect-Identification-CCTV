# Suspect Identification in CCTV Footage

## Dataset

The dataset used for this project is [ChokePoint](https://arma.sourceforge.net/chokepoint/), which consists of 25 subjects (19 male and 6 female) in portal 1 and 29 subjects (23 male and 6 female) in portal 2. The recording of portal 1 and portal 2 are one month apart. The dataset has frame rate of 30 fps and the image resolution is 800X600 pixels.

In total, the dataset consists of 48 video sequences and 64,204 face images. In all sequences, only one subject is presented in the image at a time. The first 100 frames of each sequence are for background modelling where no foreground objects were presented.

## Setup

To run the project, you need to install the conda environment from the `environment.yml` file.

```bash
conda env create -f environment.yml
```

To download the dataset, you need to run the `dataset.py` script. This process will take a while. It will download the dataset, save all the archives in the `downloaded` folder and then extract everything in the `data` folder.

```bash
python dataset.py
```
