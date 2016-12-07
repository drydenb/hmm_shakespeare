import pprint
import sys
import enchant

from data_parser import parser

def flatten(lst):

	"""
	Flattens a list. Takes a list of lists such as [ [[],[],[]], [[],[],[]] ]
	and returns [[], [], [], [], [], []]
	"""

	return [item for sublist in lst for item in sublist]

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

def get_hashes(data):

	"""
	From the flattened list of filtered data, create two hashes, one that maps 
	a word to an id and one that maps the id back to the word. 
	"""

	# flatten the flattened data once again to get a list of all tokens
	tokens = flatten(data)

	# generate the hashes
	unique_tokens = set(tokens)
	id_to_token = dict(enumerate(unique_tokens))
	token_to_id = {v : k for k,v in id_to_token.iteritems()}

	return id_to_token, token_to_id

def create_training(lines, token_to_id):

	"""
	Takes in the flattened filtered data. Creates the training samples for the 
	Baum-Welch algorithm. 
	"""

	for idx, line in enumerate(lines):
		lines[idx] = map(lambda t: token_to_id[t],line)
	return lines

result = get_data('./data/raw/shakespeare.txt')
filtered = filter_data(result)
flattened = flatten(filtered)

id_to_token, token_to_id = get_hashes(flattened)
training = create_training(flattened, token_to_id)

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(training)
# print len(flattened)


# pp.pprint(id_to_token)
# pp.pprint(token_to_id)
