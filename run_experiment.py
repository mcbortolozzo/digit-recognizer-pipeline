from sklearn.preprocessing import MinMaxScaler
from itertools import product

import argparse
import pandas as pd
import numpy as np
import yaml
import logging
import json

from models import *
import utils

logger = logging.getLogger('Digit Recognition Model Optimizer')
logger.setLevel(logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console)

parser = argparse.ArgumentParser()
#parser.add_argument('--model', type=str, help='btc, cnn, crnn', default='btc')
parser.add_argument('--train_data', type=str, help='Train Dataset', default='data/train.csv')
parser.add_argument('--prediction_data', type=str, help='Dataset to Predict', default='data/test.csv')
parser.add_argument('--run_config', type=str, help='Run Configuration', default='run_config.yaml')
#parser.add_argument('--early_stop', type=bool, help='no improvement during 10 epoch -> stop', default=True)
args = parser.parse_args()

# Load configuration file for pipeline
with open(args.run_config) as f:
	config = yaml.load(f, Loader=yaml.Loader)

# Load Available models
models = utils.get_models()
logger.info('Available models %s' % list(models.keys()))


# Load and normalize data
input_data = pd.read_csv(args.train_data)
input_data.iloc[:, 1:] = MinMaxScaler().fit_transform(input_data.iloc[:, 1:])
train, validate, test = np.split(input_data.sample(frac=1), [int(.6*len(input_data)), int(.8*len(input_data))])

train_Y = train.iloc[:, 0].values
train_X = train.iloc[:, 1:].values

validation_Y = validate.iloc[:, 0].values
validation_X = validate.iloc[:, 1:].values

validation_scores = {}

for model in config:
	if model not in models:
		logger.error('Model %s not found, skipping' % model)
		continue
	else:
		logger.info('Running training and optimization for Model %s' % model)
		model_config = config[model]
		config_combinations = list(product(*model_config.values()))
		logger.info('Total of Parameter Combinations being tested: %d' % len(config_combinations))

		model_scores = {}
		for comb in config_combinations:
			model_args = utils.join_model_args(list(model_config.keys()), comb)
			logger.info('Running for Parameters: %s' % model_args)
			classifier = models[model](model_args)
			logger.info('Training model')
			classifier.train_model(train_X, train_Y)
			logger.info('Estimating dev set')
			score = classifier.score(validation_X, validation_Y)
			logger.info('Score: %.2f' % score)
			model_args_string = ', '.join(model_args)
			model_scores[model_args_string] = {'model': classifier, 'score':score, 'parameters': model_args}

		sorted_scores = sorted(model_scores.items(), key= lambda x: x[1]['score'])
		best_model_score = sorted_scores[0][1]['score']
		best_model_args = sorted_scores[0][0]

		logger.info('Best %s result: %.2f with args: %s' % (model, best_model_score, best_model_args))

		validation_scores[model] = model_scores




print(json.dumps(validation_scores, indent=4))


