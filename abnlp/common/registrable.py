"""
modified from: https://github.com/allenai/allennlp/blob/master/allennlp/common/registrable.py
"""

from collections import defaultdict

class Registrable(object):

	_registry = defaultdict(dict)
	default_implementation = None

	@classmethod
	def register(cls, name):
		registry = Registrable._registry[cls]
		def add_subclass_to_registry(subclass):
			assert name not in registry
			registry[name] = subclass
			return subclass
		return add_subclass_to_registry

	@classmethod
	def by_name(cls, name):
		assert name in Registrable._registry[cls]
		return Registrable._registry[cls].get(name)
