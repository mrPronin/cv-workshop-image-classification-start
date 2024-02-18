# Image classification

Current repository is based on [Image Classification for beginner](https://medium.com/mlearning-ai/image-classification-for-beginner-a6de7a69bc78) tutorial (with some refactoring).

This repository will also be the basis for the "Workshop 1. Computer Vision | Image classification".

## How to setup

1. Install [Miniconda](https://docs.anaconda.com/free/miniconda/) for your operation system. Current repository using conda 24.1.0
2. Clone [this repository](https://github.com/mrPronin/ml-image-classification-for-beginner) to your workshop folder
3. Run next command to create an environment from an environment.yml file

```bash
conda env create -y -f environment.yml
```
Verify that the new environment was installed correctly:
```bash
conda env list
```

4. Activate a new environment
```bash
conda activate cv-workshop-01-3_11
```

5. Create `data` subfolder in the project folder. Download the data set - [Rice Image Dataset](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset/data) and place it in the `data` folder.

6. Train the model
```python
python ./src/image_classification_for_beginner/main.py
```

7. Test the model with selected image
```python
python ./src/image_classification_for_beginner/test_model.py
```