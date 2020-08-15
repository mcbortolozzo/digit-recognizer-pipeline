
class SklearnBase():

	def __init__(self, args = {}):
		self.args = args
		self.parse_args(args)
		self.classifier = self.build_model()

	def train_model(self, train_X, train_Y):
		self.model = self.classifier.fit(train_X, train_Y)
		return self.model

	def score(self, test_X, test_Y):
		return self.model.score(test_X, test_Y)