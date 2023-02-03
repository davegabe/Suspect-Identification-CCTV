# Suspect Identification in CCTV Footage

## Dataset

The dataset used for this project is [ChokePoint](https://arma.sourceforge.net/chokepoint/), which consists of 25 subjects (19 male and 6 female) in portal 1 and 29 subjects (23 male and 6 female) in portal 2. The recording of portal 1 and portal 2 are one month apart. The dataset has frame rate of 30 fps and the image resolution is 800X600 pixels.

In total, the dataset consists of 48 video sequences and 64,204 face images. In all sequences, only one subject is presented in the image at a time. The first 100 frames of each sequence are for background modelling where no foreground objects were presented.

## Setup

To run the project, you need to install the conda environment from the `environment.yml` file.

```bash
# Create the conda environment
conda env create -f environment.yml
# Activate the conda environment
conda active si-cctv
# Add LD_LIBRARY_PATH to the conda environment
conda env config vars set LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
# Add XLA_FLAGS to the conda environment
conda env config vars set XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda
```

To download the dataset, you need to run the `dataset.py` script. This process will take a while. It will download the dataset, save all the archives in the `downloaded` folder and then extract everything in the `data` folder.

```bash
python dataset/dataset.py
```

## Running the project

To run the project, you need to run the `main.py` script. Use the following command to run the project.

```bash
python -m main
```