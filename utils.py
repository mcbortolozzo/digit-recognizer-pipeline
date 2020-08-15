#import pkgutil
import inspect

MODEL_MODULE = "models"

def get_models():
	#loader = pkgutil.get_loader(MODEL_MODULE)
	models = {}
	module = __import__(MODEL_MODULE)
	for element_name in dir(module):
		element = getattr(module, element_name)
		if inspect.isclass(element) and hasattr(element, "Name"):
			model_name = getattr(element, "Name")()
			models[model_name] = element
	return models

def join_model_args(args_names, arg_values):
	args = {}
	for i in range(len(args_names)):
		args[args_names[i]] = arg_values[i]

	return args