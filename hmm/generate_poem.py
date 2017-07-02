#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Developed by: Dryden Bouamalay
# Purpose: Functions for generating poems using a trained HMM


import random
import numpy as np 


def choose_idx(arr):
    """Choose an index of the given array arr using its entries as weights.
    """
    init = random.uniform(0, sum(arr))
    upper = 0
    for idx, a in enumerate(arr):
        upper += a
        if upper > init:
            return idx
    raise IndexError('Failed to generate valid index for array')


def seed(S):
    """Choose an initial state to seed the generation. Use the starting
    distribution S utilized by the Baum-Welch algorithm. 
    """
    return choose_idx(S)


def next_hidden(s, A):
    """From a given state s, use the transition matrix A to generate the next
    hidden state. 
    """
    return choose_idx(A[s])


def emit(s, O):
    """Given a hidden state s and the observation matrix O, emit an observation
    possible from state s by using the probabilities in O[s]."""
    return choose_idx(O[s])


def map_to_words(poem, id_to_token):
    """Convert the poem we generated from word ids back to English words."""
    return [map(lambda w: id_to_token[w], line) for line in poem]


def pretty_print_poem(poem):
    """Pretty prints a poem to STDOUT."""
    for line in poem:
        print " ".join(line)


def generate_line(state, line_length, A, O):
    """Given a current hidden state, generate a new line by emitting an
    observation and then transitioning to a new hidden state. Repeat until the
    line is as long as the line_length parameter.
    """
    line = []
    for _ in range(line_length):
        line.append(emit(state, O))
        state = next_hidden(state, A)

    return state, line


def generate_poem(mu, sigma, num_lines, A, O, S):
    """Given a mean line length mu and standard deviation sigma, generate a
    poem consisting of num_lines by calling generate_line().
    """
    poem = []
    state = seed(S)
    for _ in range(num_lines):
        line_length = int(np.random.normal(mu, sigma, 1)[0])
        new_state, line = generate_line(state, line_length, A, O)
        state = new_state
        poem.append(line)
    return poem
