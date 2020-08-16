import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiplicativeLR
from torch.utils.data import TensorDataset, DataLoader
import copy

import numpy as np

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNNModel():

	def __init__(self, args = {}):
		self.args = args
		self.parse_args(args)
		self.classifier = ConvNet()
		self.optimizer = optim.Adam(self.classifier.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-9)
		self.loss_function = nn.CrossEntropyLoss()
		lmbda = lambda epoch: self.lr_factor
		self.lr_scheduler = MultiplicativeLR(self.optimizer, lr_lambda=lmbda)

	def parse_args(self, args):
		self.lr = args['learning_rate'] if 'learning_rate' in args else 0.001
		self.max_epoch = args['max_epoch'] if 'max_epoch' in args else 100
		self.early_stop = args['early_stop'] if 'early_stop' in args else False
		self.batch_size = args['batch_size'] if 'batch_size' in args else 64
		self.shuffle = args['shuffle'] if 'shuffle' in args else False
		self.adjust_lr = args['adaptive_learning_rate'] if 'adaptive_learning_rate' in args else False
		self.early_stop_idx_limit = 10
		self.lr_factor = 0.95
		self.min_lr = 5e-6

	def adjust_learning_rate(optimizer, factor=.5, min_lr=0.00001):
	    for i, param_group in enumerate(optimizer.param_groups):
	        old_lr = float(param_group['lr'])
	        new_lr = max(old_lr * factor, min_lr)
	        param_group['lr'] = new_lr
	        logger.info('adjusting learning rate from %.6f to %.6f' % (old_lr, new_lr))

	def train_model(self, train_X, train_Y):
		if self.early_stop:
			best_acc = 0
			best_model = None
			early_stop_idx = 0

			train_X, dev_X = np.split(train_X, [int(len(train_X)*.8)])
			train_Y, dev_Y = np.split(train_Y, [int(len(train_Y)*.8)])

			tensor_dev_X = torch.Tensor(dev_X)
			tensor_dev_Y = torch.Tensor(dev_Y).type(torch.LongTensor)
			dev = TensorDataset(tensor_dev_X, tensor_dev_Y)
			dev_loader = DataLoader(dev, batch_size=self.batch_size, shuffle=False)

		tensor_train_X = torch.Tensor(train_X)
		tensor_train_Y = torch.Tensor(train_Y).type(torch.LongTensor)
		train = TensorDataset(tensor_train_X, tensor_train_Y)
		train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=self.shuffle)
		prev_loss = np.inf

		for epoch in range(self.max_epoch):
			running_loss = 0.0
			for i, data in enumerate(train_loader):
				features, labels = data
				self.optimizer.zero_grad()
				outputs = self.classifier(features.view(features.size(0), 1, 28, 28))
				loss = self.loss_function(outputs, labels)
				loss.backward()
				self.optimizer.step()

				running_loss += loss.item()

			print("epoch: ", epoch, "training loss: ", running_loss)

			if self.adjust_lr and running_loss > prev_loss:
				old_lr = self.optimizer.param_groups[0]['lr']
				self.lr_scheduler.step()
				new_lr = self.optimizer.param_groups[0]['lr']
				print("Adjusting learning rate from %.5f to %.5f" % (old_lr, new_lr))

			prev_loss = running_loss

			if self.early_stop:
				with torch.no_grad():
					dev_correct = 0.
					dev_total = 0.
					dev_loss = 0.
					for data in dev_loader:
						features, labels = data
						outputs = self.classifier(features.view(features.size(0), 1, 28, 28))
						loss = self.loss_function(outputs, labels)
						_, predicted = torch.max(outputs.data, 1)
						dev_total += labels.size(0)
						dev_correct += (predicted == labels).sum().item()
						dev_loss += loss.item()

					current_acc = dev_correct/dev_total

					if current_acc > best_acc:
						print("Best dev accuracy obtained: %.3f" % current_acc)
						best_model = copy.deepcopy(self.classifier)
						best_acc = current_acc
						early_stop_idx = 0
					else:
						early_stop_idx += 1

				if early_stop_idx >= self.early_stop_idx_limit:
					print("early stop triggered")
					self.classifier = best_model
					break

			


		return self

	def score(self, test_X, test_Y):
		tensor_test_X = torch.Tensor(test_X)
		tensor_test_Y = torch.Tensor(test_Y).type(torch.LongTensor)
		test = TensorDataset(tensor_test_X, tensor_test_Y)
		test_loader = DataLoader(test, batch_size=self.batch_size, shuffle=False)
		correct = 0.0
		total = 0.0
		with torch.no_grad():
			for data in test_loader:
				features, labels = data
				outputs = self.classifier(features.view(features.size(0), 1, 28, 28))
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()

		return correct/total


	@staticmethod
	def Name():
		return "CNN"