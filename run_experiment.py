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
parser.add_argument('--runs', type=int, help='Number of Runs for each model', default=1)
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

test_Y = test.iloc[:, 0].values
test_X = test.iloc[:, 1:].values

validation_scores = {}
best_model_configs = []

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

			average_score = 0
			best_model = None
			for i in range(args.runs):		
				logger.info("Run Number %d" % (i+1))		
				classifier = models[model](model_args)
				logger.info('Training model')
				classifier.train_model(train_X, train_Y)
				logger.info('Estimating dev set')
				score = classifier.score(validation_X, validation_Y)
				logger.info('Score: %.2f' % score)
				
				average_score += score

				if best_model is None or best_model['validation_score'] < score:
					best_model = {'classifier': classifier, 'validation_score':score, 'parameters': model_args, 'type': model}

			average_score /= args.runs
			best_model['average_validation_score'] = average_score
			logger.info("Average validation score: %.2f" % (average_score))
			logger.info('---------------------------------------------------')

			model_args_string = ', '.join(model_args)
			model_scores[model_args_string] = best_model

		sorted_scores = sorted(model_scores.items(), key= lambda x: x[1]['average_validation_score'], reverse=True)
		best_model_score = sorted_scores[0][1]['average_validation_score']
		best_model_args = sorted_scores[0][0]

		logger.info('Best %s result: %.2f with args: %s' % (model, best_model_score, best_model_args))
		logger.info('===================================================')
		best_model_configs.append(sorted_scores[0][1])
		validation_scores[model] = model_scores


for model in best_model_configs:
	logger.info("Running Test set for %s" % model['type'])
	model['test_score'] = model['classifier'].score(test_X, test_Y)
	logger.info("Test score for %s model: %.2f" % (model['type'], model['test_score']))

sorted_test_scores = sorted(best_model_configs, key=lambda x: x['test_score'], reverse=True)

logger.info("Final Model Test Scores:")
for model in sorted_test_scores:
	logger.info("%s - %.2f" % (model['type'], model['test_score']))