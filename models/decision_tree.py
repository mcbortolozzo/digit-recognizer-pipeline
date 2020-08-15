from sklearn.tree import DecisionTreeClassifier

from .sklearn_base import SklearnBase

class DecisionTreeModel(SklearnBase):

	def build_model(self):
		return DecisionTreeClassifier(criterion=self.criterion, min_samples_split=self.min_samples_split, max_features=self.max_features)

	def parse_args(self, args):
		self.criterion = args['criterion'] if 'criterion' in args else 'gini'
		self.min_samples_split = args['min_samples_split'] if 'min_samples_split' in args else 2
		self.max_features = args['max_features'] if 'max_features' in args else None

	@staticmethod
	def Name():
		return 'DecisionTree'
