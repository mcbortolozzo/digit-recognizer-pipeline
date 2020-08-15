from sklearn.svm import SVC

from .sklearn_base import SklearnBase

class SVCModel(SklearnBase):

	def build_model(self):
		return SVC(C=self.C, kernel=self.kernel)

	def parse_args(self, args):
		self.C = args["C"] if "C" in args else 1
		self.kernel = args["kernel"] if "kernel" in args else "rbf"

	@staticmethod
	def Name():
		return "SVC"
