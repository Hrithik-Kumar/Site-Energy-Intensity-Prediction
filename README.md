# Site-Energy-Intensity-Prediction

## Dataset Description

According to a [report](https://www.iea.org/reports/tracking-buildings-2021) issued by the International Energy Agency (IEA), the lifecycle of buildings from construction to demolition was responsible for 37% of global energy-related and process-related CO2 emissions in 2020. Yet it is possible to drastically reduce the energy consumption of buildings by a combination of easy-to-implement fixes and state-of-the-art strategies. 

The dataset consists of building characteristics, weather data for the location of the building, as well as the energy usage for the building, and the given year, measured as Site Energy Usage Intensity (Site EUI). Each row in the data corresponds to a single building observed in a given year.

## Problem Statement

Two datasets are provided: (1) the train_dataset where the observed values of the Site EUI for each row are provided and (2) the x_test dataset the observed values of the Site EUI for each row are removed and provided separately in y_test. The task is to predict the Site EUI for each row (using the complete training dataset), given the characteristics of the building and the weather data for the location of the building. Use the test sets for validation and testing. 

The target variable is **site_eui** and the evaluation metric is **RMSE** score.

## Install

This project requires **Python** and the following Python Libraries installed:

### Notebook

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [matplotlib](http://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [plotly](https://plotly.com/)
- [imblearn](https://imbalanced-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [CatBoost](https://catboost.ai/)

### Code

Python Notebook is provided as the `notebook.ipynb` file. The required datasets are included in the `data` Folder. 

### Running the Python Notebook Locally

First, clone the repository. Then, in a terminal or command window, navigate to the top-level project directory `Site-Energy-Intensity-Prediction/` (that contains this README) and then run one of the following commands:

```bash
ipython notebook "Notebook_name"
```  
or
```bash
jupyter notebook "Notebook_name"
```

### Testing the Model Locally

To test the model locally, ensure you have [Flask](https://flask.palletsprojects.com/) installed. After a successful installation of Flask, follow these steps:

1. Open one terminal window and run the following command:
    ```bash
    python predict.py
    ```

2. Open another terminal window and execute the following command:
    ```bash
    python predict-test.py
    ```

These commands will allow you to send requests to the model and observe the results. Make sure both terminals are active while testing the model locally.


