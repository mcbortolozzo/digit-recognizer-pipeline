from sklearn.svm import SVC


class SVCModel():

	def __init__(self, args = {}):
		self.args = args
		self.parse_args(args)
		self.classifier = SVC(C=self.C, kernel=self.kernel)

	def parse_args(self, args):
		self.C = args["C"] if "C" in args else 1
		self.kernel = args["kernel"] if "kernel" in args else "rbf"

	def train_model(self, train_X, train_Y):
		self.model = self.classifier.fit(train_X, train_Y)
		return self.model

	def score(self, test_X, test_Y):
		return self.model.score(test_X, test_Y)

	@staticmethod
	def Name():
		return "SVC"
