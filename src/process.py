import pprint
import enchant

from data_parser import parser

def get_data(filename):
	
	"""
	Takes a filename containing raw poem data and produces a list of poems,
	each of which is a list of lines containing tokens.
	"""

	try:
		with open(filename, 'r') as datafile:
			data = datafile.read()
			return parser.parse(data)
	except IOError:
		print "Unable to read data file:", filename
		sys.exit(1) 

def filter_data(data):
	
	"""
	Take the list of poems containing lists of tokens and throw out any token 
	that isn't an English word.
	"""

	en_dict = enchant.Dict("en_US")
	for idx, d in enumerate(data):
		data[idx] = map(lambda l: [w for w in l if en_dict.check(w)], d)
	return data

def flatten_data(data):

	"""
	Take the list of poems containing lists of lines and flatten it such that
	we only have a list of lines (one list of every line for all poems). 
	"""

	return [line for poem in data for line in poem]

result = get_data('./data/raw/shakespeare.txt')
filtered = filter_data(result)
flattened = flatten_data(filtered)

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(flattened)
