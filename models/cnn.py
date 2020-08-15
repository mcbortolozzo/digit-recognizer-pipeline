import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

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

	def parse_args(self, args):
		self.lr = 0.001
		self.epochs = 50
		self.batch_size = 64
		self.shuffle = False

	def train_model(self, train_X, train_Y):
		tensor_train_X = torch.Tensor(train_X)
		tensor_train_Y = torch.Tensor(train_Y).type(torch.LongTensor)
		train = TensorDataset(tensor_train_X, tensor_train_Y)
		train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=self.shuffle)

		for epoch in range(self.epochs):
			running_loss = 0.0
			for i, data in enumerate(train_loader):
				features, labels = data
				self.optimizer.zero_grad()
				outputs = self.classifier(features.view(features.size(0), 1, 28, 28))
				loss = self.loss_function(outputs, labels)
				loss.backward()
				self.optimizer.step()

				running_loss += loss.item()

			print("epoch: ", epoch, "loss: ", running_loss)

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