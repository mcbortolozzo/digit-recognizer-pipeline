# Example Pipeline for Kaggle Digit Recognition Problem

The presented repository provides a simple pipeline for the Kaggle Digit Recognition dataset, but which could be adapted for other data without much effort

It provides a few algorithms already integrated, but could be easily extended for others.

## Installation

```
pip install -r requirements.txt

```

## Usage

To obtain a full log of the results generate a yaml file with the required configuration, as the provided example under "run_config.yaml"

Inform the training data, which will automatically be partitioned with the --train_data parameter

```
python run_experiment.py --train_data path_to_data_csv --run_config run_config.yaml

```

Besides, another dataset could be provided for prediction only, with the selected optimal model

```
python run_experiment.py --train_data path_to_data_csv --run_config run_config.yaml --prediction_data path_to_prediction_csv

```