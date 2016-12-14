#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Developed by: Dryden Bouamalay
# Purpose: To train an HMM and generate a poem from start to finish

import os
import sys
import argparse

from time import gmtime, strftime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 
                                                '..')))
from parse import process
from generate import generate_poem
from bw import baum_welch

def main(argv):

    """
    After parsing the number of desired hidden states & the tolerance from 
    the command line, train a hidden markov model on the poems using the raw
    dataset. After the model is trained, use it to generate a poem from a 
    random seed sampled from the starting distribution of the model.
    """

    ########################################
    # Parse command line arguments and initialize
    ########################################

    parser = argparse.ArgumentParser(
        description="Trains a hidden markov model using poetry data.")
    parser.add_argument('-s', '--states',
        type=int,
        help='The number of hidden states for the model.',
        required=True)
    parser.add_argument('-t', '--tolerance',
        type=float,
        help='The difference at which we stop convergence ' 
             'between the norms of the transition matrix A and '
             'the observation matrix O.',
        required=True)

    args = parser.parse_args()

    num_states = args.states
    tolerance = args.tolerance 
    id_pickle = strftime("%Y-%m-%d-%H:%M:%S", gmtime())

    src_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(src_dir, '..', 'data')
    shakespeare_file = os.path.join(data_dir, 'shakespeare.txt')

    (
        num_obs, 
        id_to_token, 
        token_to_id, 
        obs
    ) = process.process_data(shakespeare_file)

    ########################################
    # Train the HMM
    ########################################

    print "Number of samples in dataset is: ", len(obs), "\n"

    # sanity check that no index in the dataset is >= num_obs or < 0.
    assert baum_welch.check_obs(num_obs, obs) == True    

    # output the mean and std_dev for this observation sequence with ID 
    mean, std = baum_welch.find_mean_std(obs)
    
    # perform training on the list of observations obs 
    S, A, O = baum_welch.baum_welch(num_states, num_obs, obs, tolerance) 

    print "Final transition matrix A: \n", A, "\n"
    print "Final observation matrix O: \n", O, "\n"

    ########################################
    # Generate a sample poem
    ########################################

    SONNET_LINES = 14 

    print "Generated poem:"
    id_poem = generate_poem.generate_poem(mean, std, SONNET_LINES, A, O, S)
    poem = generate_poem.map_to_words(id_poem, id_to_token)
    generate_poem.pretty_print_poem(poem)

    sys.exit()

################################################################################
# MAIN
################################################################################

if __name__ == '__main__':
    main(sys.argv)