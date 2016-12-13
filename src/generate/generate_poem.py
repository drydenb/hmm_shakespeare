
import sys
import random 
import numpy as np 

def choose_idx(arr):

	"""
	Assume that arr is an array containing probabilities. Choose an index of
	this array that uses these probabilities as weights.
	"""

	init = random.uniform(0, 1)
	upper = 0
	for idx, a in enumerate(arr):
		upper += a
		if upper > init:
			return idx
	raise IndexError('Failed to generate valid index for array')

def seed(S):

	"""
	Choose an initial state to seed the generation. Use the starting 
	distribution S utilized by the Baum-Welch algorithm. 
	"""

	return choose_idx(S)

def next_hidden(s, A):

	"""
	From a given state s, use the transition matrix A to generate the next 
	hidden state. 
	"""

	# TODO: Fails frequently
	return choose_idx(A[s])

def emit(s, O):

	"""
	Given a hidden state s and the observation matrix O, emit an observation 
	possible from state s by using the probabilities in O[s].
	"""

	return choose_idx(O[s])


def generate_line(state, line_length, A, O):

	""" 
	"""

	line = []
	for _ in range(line_length):
		line.append(emit(state, O))
		state = next_hidden(state, A)

	return state, line

def generate_poem(mu, sigma, num_lines, A, O, S):

	"""
	"""

	poem = []
	state = seed(S) 

	for _ in range(num_lines):
		line_length = int(np.random.normal(mu, sigma, 1)[0])
		new_state, line = generate_line(state, line_length, A, O)
		state = new_state
		poem.append(line)

	return poem
