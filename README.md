# Feature-Optimization-for-Musk-Molecule-Classification-Using-GA-and-PSO
# Feature Selection and Machine Learning Model Comparison

This repository contains Python code for comparing the performance of different machine learning models with and without feature selection. The code uses the Musk dataset to evaluate the impact of feature selection on model accuracy.

## Models Implemented

The following machine learning models are implemented:

* Random Forest Classifier
* K-Nearest Neighbors (KNN) Classifier
* Support Vector Machine (SVM) Classifier

## Feature Selection Methods

The following feature selection methods are used:

* **Genetic Algorithm (GA):** A metaheuristic optimization algorithm inspired by the process of natural selection.
* **Particle Swarm Optimization (PSO):** A computational method that optimizes a problem by iteratively trying to improve a candidate solution with regard to a given measure of quality.

## Code Description

The code performs the following steps:

1.  **Data Loading:** Loads the Musk dataset from a CSV file (`musk.csv`).
2.  **Data Preprocessing:**
    * Separates the features (X) and the target variable (y).
    * Scales the features using MinMaxScaler to normalize them between 0 and 1.
    * Splits the data into training and testing sets (75% train, 25% test).
3.  **Model Training and Evaluation:**
    * Trains each of the machine learning models (Random Forest, KNN, SVM) using all features.
    * Applies the Genetic Algorithm (GA) and Particle Swarm Optimization (PSO) for feature selection.
    * Trains each model using the features selected by GA and PSO.
    * Evaluates the performance of all models using accuracy, precision, recall, and F1-score.
    * Plots the confusion matrix for each model.
4.  **Performance Comparison:**
    * Compares the performance of the models with and without feature selection using bar plots.

## Libraries Used

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
    * `train_test_split`
    * `MinMaxScaler`
    * `RandomForestClassifier`
    * `KNeighborsClassifier`
    * `SVC`
    * `accuracy_score`
    * `precision_score`
    * `recall_score`
    * `f1_score`
    * `classification_report`
    * `confusion_matrix`
* random

## How to Run the Code

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
2.  **Install the required libraries:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```
3.  **Download the dataset:**
    * Download the `musk.csv` dataset and place it in the same directory as the Python scripts.
4.  **Run the Python scripts:**
    * Run each of the Python scripts (e.g., `random_forest.py`, `knn.py`, `svm.py`) to train and evaluate the models.
    ```bash
    python random_forest.py
    python knn.py
    python svm.py
    ```

## Results

The code will output the performance metrics (accuracy, precision, recall, F1-score) and display the confusion matrix for each model. It will also generate bar plots comparing the performance of the models with and without feature selection.

## Contributions

Contributions to this repository are welcome. Feel free to submit pull requests or open issues to suggest improvements or report bugs.
