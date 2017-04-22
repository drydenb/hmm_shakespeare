#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Developed by: Dryden Bouamalay
# Purpose: Implements functions for processing data after parsing


import os
import enchant
from data_parser import parser


def flatten(lst):

    """
    Flattens a list. i.e., takes a list of lists such as 
    [ [[],[],[]], [[],[],[]] ] and returns [ [], [], [], [], [], [] ]
    """

    return [item for sublist in lst for item in sublist]


def get_data():
    
    """
    Takes a filename containing raw poem data and produces a list of poems,
    each of which is a list of lines containing tokens.
    """

    # try:
    file_dir = os.path.dirname(os.path.realpath(__file__))
    resources_dir = os.path.join(file_dir, 'resources')
    shakespeare_file = os.path.join(resources_dir, 'shakespeare.txt')

    with open(shakespeare_file, 'r') as datafile:
        data = datafile.read()
        return parser.parse(data)

    # except IOError:
    #   print "Unable to read data file:", filename
    #   sys.exit(1)


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

    # remove duplicate tokens
    unique_tokens = set(tokens)
    num_unique = len(unique_tokens)

    # generate the hashes
    id_to_token = dict(enumerate(unique_tokens))
    token_to_id = {v : k for k,v in id_to_token.iteritems()}

    return num_unique, id_to_token, token_to_id


def create_training(lines, token_to_id):

    """
    Takes in the flattened filtered data. Creates the training samples for the 
    Baum-Welch algorithm. 
    """

    for idx, line in enumerate(lines):
        lines[idx] = map(lambda t: token_to_id[t],line)
    return lines


def process_data():

    """
    Takes in the raw poem data filename, calculates the hashes needed, 
    and returns them along with the number of unique words in the dataset.
    """

    raw = get_data()
    filtered = filter_data(raw)
    flattened = flatten(filtered)
    num_unique, id_to_token, token_to_id = get_hashes(flattened)
    train_data = create_training(flattened, token_to_id)

    return num_unique, id_to_token, token_to_id, train_data
