# Dog-Breed-Classification
This repository consists of implementation of Dog Breed Classifier using ResNet 50.

## Project Directory Structure

```
  ├── Dog-Breed-Classification              # Root Folder
    ├── model/                              # Folder contains the saved keras model
    ├── .gitignore                          # .gitignore file
    ├── Dog_Breed_Classification.ipynb      # Python notebook which contains dog breed classfication using ResNet50
    ├── Procfile                            # Procfile file for heroku deployment
    ├── README.MD                           # Readme file
    ├── Resnet50_Scratch.ipynb              # Python notebook which contains dog breed classfication using ResNet50 from scratch
    ├── api.py                              # Python File which contains the REST API Code
    ├── inference.py                        # Python File which contains the Model Inference Code
    ├── requirements.txt                    # Python requirements file
```

## Dataset
The dataset used for this project is not included in this repository and can be obtained from the following link
> Link: https://www.kaggle.com/c/dog-breed-identification/data

Download the dataset and place it inside a folder called 'data' inside the project repository.

## Installation

1. Clone the repository into a folder
```
$ git clone https://github.com/kishore-s-15/Dog-Breed-Classification.git
```

2. Create a virtual environment
```
python -m venv .venv (or) python3 -m venv .venv
```

3. Activate the virtual environment

   > On Windows run
   ```
   .venv\Scripts\activate.bat
   ```
   
   > On Linux and MacOs run
   ```
   source .venv/bin/activate
   ```
   
4. Install the dependencies for the project in the virtual environment
```
pip install -r requirements.txt
```
   
5. Then run the following command
```
uvicorn api:app --reload
```
This should start a web server running at your localhost:8000
