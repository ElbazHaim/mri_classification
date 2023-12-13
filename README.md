## MRI Alzheimer's Dementia Stage Classifier

### Overview
This project is a deep learning endeavor focused on Alzheimer's image classification with four classes. It leverages PyTorch Lightning for streamlined training and Optuna for hyperparameter tuning.

Currently only LeNet is implemented.


### Project Structure
The project is organized as follows:

- `tune.py`: Serves as the main training and tuning script. Orchestrates training and hyperparameter optimization using Optuna.

- `models/`: Houses the deep neural network model implementations for the image classification. 

- `datamodules/`: Contains the PyTorch Lightning datamodule which handles the data itself.

- `transformations/`: Torchvision transformations for image preprocessing.

- `Makefile`: Simplifies the execution of common tasks.

- `requirements.txt`: List all the necessary dependencies.

- `.gitignore`: Specifies which files and directories should be ignored when committing to version control.

- `README.md`: The current document :blush:


### Usage
To use this project:

1. Clone or download the repository to your local machine.

2. Install the required dependencies by using `pip` with the packages listed in the `requirements.txt` file, or use the makefile 

```
make install
```


5. Explore and edit the `tune.py` script to start training and optimizing. Also usable as 
```
make tune
```

### Credits
Data source: 

https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images

