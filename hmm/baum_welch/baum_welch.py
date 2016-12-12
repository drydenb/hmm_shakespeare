from __future__ import division

import os 
import sys 
import csv 
import getopt
import random 
import pickle 
import numpy as np 

from tempfile import TemporaryFile
from scipy.integrate import quad
from scipy.linalg import norm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from parser import process
# print sys.path

################################################################################
# FUNCTION DEFINITIONS
################################################################################


def usage(argv):

    """
    Prints the usage statement for the script. 
    """

    print ("usage: " + 
            str(argv[0]) +
            " -p <pickle_file> -n <num_obs> -s <num_states> -t <tolerance> " + 
            "-o <id_num>")
    print """
          Options:
          -p [ --obs_pickle ]    The name of the pickle file storing the list 
                                 of observations 
          -n [ --num_obs ]       The number of distinct observations possible 
                                 for the list of observations, i.e. 3232 means 
                                 valid indices are from 0 to 3231 
          -s [ --num_states ]    The number of hidden states to use for the HMM.
          -t [ --tolerance ]     The difference at which we stop convergence 
                                 between the norms of the transition and 
                                 observation matrices A and O.   
          -o [ --id_pickle ]     The number to identify this HMM. 

          Example: 
          python baum_welch.py ./pickles/sonnet_to_index.p 3232 10 0.01 
          """
    return 


# transition matrix has rows as target state and columns as start state 
def forward(S, A, O, obs):
    """ 
    Calculates the forward probability matrix F. This is a matrix where each 
    (i, j) entry represents P(o_1, o_2, ... o_j, X_t = i| A, O). In other words, 
    each (i, j) entry is the probability that the observed sequence is o_1, ... 
    o_j and that at position j we are in hidden state i. We build F from the 
    first observation o_1 up to the entire observed sequence o_1, ... o_M. Thus 
    F has dimension L x M where L is the number of hidden states and M is the 
    length of our input sample 'obs'. 

    @params: 
    S    np.array - state vector for starting distribution.

    A    np.array - transition matrix, L x L for L hidden states, each (i, j) 
                    entry is P(X_i | X_j), or the probability of transitioning 
                    from start state X_j (column entry) to target state X_i 
                    (row entry).

    O    np.array - observation matrix, L x M' for L hidden states and M' total
                    possible observations. each (i, j) entry is P(Y_j | X_i), or 
                    the probability of observing observation Y_j while in state 
                    X_i.

    obs  np.array, list - the observations. these are assumed to be integers 
                          that index correctly into A and O. 
    """

    assert np.shape(A)[0] == np.shape(A)[1]    # transition matrix should be square 
    L = np.shape(A)[0]                         # L is the number of hidden states 
    M = len(obs)                               # M is the number of observations in our sample 'obs'  

    C = []                                     # the list of coefficients used to normalize each column to 1 
    F = np.zeros((L, M))                         # the foward algorithm generates an L x M matrix
    F[:,0] = np.multiply(S, O[:,obs[0]])       # initialize the first column of F via S * (obs[0] column of B)
    c_0 = np.sum(F[:,0])                       # compute the first normalizing coefficient
    C.append(c_0)                              # record c_0 
    F[:,0] = np.divide(F[:,0], c_0)            # normalize the first column so the entries sum to 1 

    # begin the forward algorithm. generate each subsequent column of F via the previous one, 
    # normalizing at each step
    for j in range(1, M):
        F[:,j] = np.dot(np.multiply(A, O[:,obs[j]]), F[:,j - 1])         # compute the new column j 
        c_j = np.sum(F[:,j])                                             # compute the jth coeff.
        C.append(c_j)                                                    # record the jth coeff.  
        F[:,j] = np.divide(F[:,j], c_j)                                  # normalize column j 

    # return the foward matrix F and the list of normalizing coefficients C (these will be used
    # to normalize the backward probabilities in the backward step)
    return (F, C) 


def backward(A, O, C, obs): 
    """ 
    Calculates the backward probability matrix B. This is a matrix where each 
    (i, j) entry represents P(o_(j + 1), o_(j + 1), ... o_M | X_t = i). Each 
    (i, j) entry is the probability that the sequence ends in o_(j + 1), 
    o_(j + 2), ... o_M where we are in hidden state i at position j. We build B 
    from the last observation o_M up to the first o_1. Thus B has dimension 
    L x M where L is the number of hidden states and M is the length of our 
    sample 'obs'. 

    params:    
    A    the transition matrix 
    O    the observation matrix 
    C   the list of forward coefficients 
    obs  the list of observations 
    """ 

    assert np.shape(A)[0] == np.shape(A)[1]    # transition matrix should be square 
    L = np.shape(A)[0]                         # L is the number of hidden states 
    M = len(obs)                               # M is the number of observations in our sample 'obs'

    B = np.zeros((L, M))                         # the backward algorithm generates an L x M matrix
    B_MM = np.ones(L)                          # the backward algorithm is initialized with all ones 

    assert len(C) == M                         # number of coeff should equal length of sequence 

    # initialize the last column of B and then normalize with the last coefficient of C 
    B[:,M - 1] = np.dot(np.multiply(A, O[:,obs[-1]]), B_MM)    
    B[:,M - 1] = np.divide(B[:,M - 1], C[-1]) 
    
    # goes from j = 2 to M, so M - j ranges from M - 2 to 0 (aka we work backwards starting with
    # the second to last column M - 2 to the first column 0)
    for j in range(2, M + 1):
        # compute the M - jth row via M - j + 1 and then normalize using C[M - j]
        B[:,M - j] = np.dot(np.multiply(A, O[:,obs[M - j]]), B[:, M - j + 1])
        B[:,M - j] = np.divide(B[:,M - j], C[M - j]) 

    return B


def gamma(S, F, B): 
    """
    Computes the gamma matrix G. This is a matrix where each (i, j) entry 
    represents gamma_j(i) = P(X_j = i | o_1, ... o_M, S, A, O). This is the 
    probability that at the jth part of our training sequence we are in hidden 
    state i. 

    params:    
    S    the starting state distribution 
    F    the forward matrix.
    B    the backward matrix.   
    """

    assert np.shape(F) == np.shape(B)       # F & B should have shape L x M 
    L = np.shape(F)[0]                      # the number of hidden states is L 
    M = np.shape(F)[1] 

    B_MM = np.ones(L)                       # recreate B_MM from backward algorithm 

    F = np.hstack((S[:,np.newaxis], F))     # add to F the vector S as its first column 
    B = np.hstack((B, B_MM[:,np.newaxis]))  # add to B the vector B_MM as its last column

    assert np.shape(F) == np.shape(B)       # F and B should still be the same size 
    G = np.multiply(F, B)                   # multiply F and B entrywise to get G

    # renormalize 
    for i in range(M):
        G[:,i] = np.divide(G[:,i], np.sum(G[:,i]))

    # now remove the first column gamma_0 such that gamma is L x M 
    G = G[:,1:]
    assert np.shape(G) == (L, M)

    return G 


def xi(A, O, S, F, B):

    """ 
    Computes the xi matrix E. This is a 3-dimensional matrix M x L x L
    """ 
    
    assert np.shape(F) == np.shape(B)       # F & B should have shape L x M 
    L = np.shape(F)[0]                      # the number of hidden states is L 
    M = np.shape(F)[1]                      # the length of the sequence is M 

    B_MM = np.ones(L)                       # recreate B_MM from backward algorithm 

    F = np.hstack((S[:,np.newaxis], F))     # add to F the vector S as its first column 
    B = np.hstack((B, B_MM[:,np.newaxis]))  # add to B the vector B_MM as its last column

    # now column F_1 correpsonds to B_1, etc.

    # initialize the 3D array 
    E = np.ones((M, L, L))

    # for every step in the M-length sequence, generate an L x L matrix as the t'th entry of our
    # M x L x L matrix 
    for t in range(M):
        t_matrix = np.ones((L, L))
        for i in range(L):
            for j in range(L): 
                t_matrix[i][j] = F[i][t] * A[i][j] * B[j][t + 1] * O[j][t + 1]
        # normalize the t_matrix so that its entries sum to 1
        if t_matrix.astype(int).all() != 0:
            t_matrix = np.divide(t_matrix, np.sum(t_matrix))
        E[t] = t_matrix 

    return E 


def difference(A, B):
    """ 
    This function compututes the difference between matrices A and O (entrywise) 
    and then returns the Frobenius norm of their difference. This acts as a 
    tolerance for our convergence condition.
    """
    T = A - B
    return norm(T)


def indicator(a, b):
    if (a == b):
        return 1
    return 0 


def baum_welch(L, M, obs, epsilon): 

    """ 
    Runs the Baum-Welch algorithm on a list of training sequences X. Returns 
    trained transition and observation matrices A and O. 

    params:    
    L          int  -  the number of hidden states to use for the HMM.
    M          int  -  the number of distinct observations possible in the 
                       training set.
    obs        list -  the list of sequences used for the training. each 
                       sequence is assumed to be a list of integers that 
                       correctly index into M.
    epsilon    float - the tolerance.
    """

    # initialize start state S 
    S = np.random.uniform(size=L)    # initialize a start state distribution S for the HMM 
    S = np.divide(S, np.sum(S))      # normalize the vector to 1 

    # initialize transition and observation matrices A and O

    # the rows of A are the target states and the columns of A are the start states. 
    # given a start state, one of the target states must be choosen so each column is normalized
    A = np.random.rand(L, L) 
    for i in range(L): 
        A[:,i] = np.divide(A[:,i], np.sum(A[:,i]))    

    # given some hidden state, there must be some observation, so every row of this matrix should
    # be normalized
    O = np.random.rand(L, M) 
    for i in range(L):
        O[i,:] = np.divide(O[i,:], np.sum(O[i,:])) 
    
    # for the moment, just do one step through the data 
    
    # initialize matrices for vectorization 
    A_numer = np.zeros(np.shape(A))
    A_denom = np.ones(L)[:,None]
    O_numer = np.zeros(np.shape(O))
    O_denom = np.ones(L)[:,None] 

    # construct a do-while loop in python  
    while True:

        A_prev = A
        O_prev = O 
        o_count = 0

        for o in obs: 

            o_count += 1 

            ####################################################################
            # E STEP
            ####################################################################

            # the length of the sample 
            Mj = len(o) 

            # perform forward and backward on the sample 
            (F, C) = forward(S, A, O, o)
            B = backward(A, O, C, o)

            # from forward and backward, compute gamma and xi for this sample 
            G = gamma(S, F, B)
            E = xi(A, O, S, F, B)

            # sanity check for dimensions of generated matrices 
            assert np.shape(G) == (L, Mj) 
            assert np.shape(E) == (Mj, L, L) 

            ####################################################################
            # M STEP 
            ####################################################################

            # TRAIN TRANSITION MATRIX 

            # populate the numerator -- this is a collapse of the 3D matrix to a 2D one with summations 
            # entrywise between each 2D array, i.e., Mj x L x L ----> L x L = shape(A)
            for m in range(Mj): 
                A_numer += E[m]

            # the denominator is the sum across Mj for row i in the L x Mj matrix G 
            A_denom += np.sum(G, axis=1)[:, None]
            assert np.shape(A_denom)[0] == L   

            # TRAIN OBSERVATION MATRIX 
            for m in range(Mj):
                # only the column concerting emission o[m] should be modified 
                O_numer[:, o[m]] += np.sum(G, axis=1)
            O_denom += np.sum(G, axis=1)[:, None]

        A = A_numer / A_denom 
        O = O_numer / O_denom 

        # renormalize 
        for i in range(L):
            A[:,i] = np.divide(A[:,i], np.sum(A[:,i]))
            O[i,:] = np.divide(O[i,:], np.sum(O[i,:]))

        # print out the difference between this iterations A, O matrices and 
        # the previous iteration's matrices 
        print "DELTA A: ", difference(A, A_prev)
        print "DELTA O: ", difference(O, O_prev)

        # convergence condition 
        if (difference(A, A_prev) < epsilon and 
            difference(O, O_prev) < epsilon):
            break

    #ENDWHILE 

    # return the trained transition and observation matrices (A, O)  
    return (S,A,O) 

def check_obs(idx, obs):
    """ 
    Checks every term in every sequence of obs and sees if any term is >= idx 
    or < 0. If true, returns false. Otherwise returns true.  
    """
    for o in obs: 
        for term in o: 
            if (term >= idx) or (term < 0):  
                return False 
    return True 

def find_mean_std(obs):
    """
    Calculates the mean and standard deviation.
    """
    lengths = [len(o) for o in obs]
    total_lines = len(lengths)
    arr_lengths = np.array(lengths)
    mean = np.mean(arr_lengths)
    std_dev = np.std(arr_lengths)
    return (mean, std_dev)

################################################################################
## MAIN
# 10 hidden states is a good testing number. The code can't handle much more 
# than 100 states. Each pass on the dataset will print out the tolerances so you 
# can see if you're converging fast enough. 
# 
# The output of A is a L x L matrix where L is the number of hidden states used. 
# the format of this matrix is structured such that the rows represent the 
# TARGET states and the columns represent the START states. so (i, j) represents 
# the probability of transitioning from the jth state to the ith state. 
#
# The output of O is a L x M matrix where L is the number of hidden states and M 
# is the number of distinct observations possible for the list of observations, 
# i.e. 10 x 3232 
################################################################################
def main(argv):

    num_unique, train_data = process.process_data('./data/raw/shakespeare.txt')
    print train_data
    print num_unique

    ########################################
    # Parse command line arguments...
    ########################################

    short_args = "hp:n:s:t:o:"
    long_args = ["help", 
                 "obs_pickle=", 
                 "num_obs=", 
                 "num_states=", 
                 "tolerance=", 
                 "id_pickle="]
    try:
        opts, args = getopt.getopt(sys.argv[1:], short_args, long_args)
    except getopt.GetoptError as err:
        print str(err)    # print the err to stdout
        usage(argv)       # print the usage statement
        sys.exit(1)

    obs_pickle = None
    num_obs = None
    num_states = None
    tolerance = None
    id_pickle = None

    try:
        for o, a in opts:
            if o in ("-h", "--help"):
                usage(argv)
                sys.exit()
            elif o in ("-p", "--obs_pickle"):
                obs_pickle = str(a) 
            elif o in ("-n", "--num_obs"):
                num_obs = int(a) 
            elif o in ("-s", "--num_states"):
                num_states = int(a)
            elif o in ("-t", "--tolerance"):
                tolerance = float(a)
            elif o in ("-o", "--id_pickle"):
                id_pickle = str(a)
            else:
                usage(argv)
                sys.exit(1)
    except ValueError:
        usage(argv)
        sys.exit(1)

    if (obs_pickle is None or
        num_obs is None or 
        num_states is None or 
        tolerance is None or 
        id_pickle is None):
        usage(argv)
        sys.exit(1)

    ########################################
    # Train the matrices...
    ########################################

    pickle_dir = "./data/pickles/"

    # unpickle the list of observations 
    obs = pickle.load(open(obs_pickle, 'rb'))
    print "Number of samples in dataset is: ", len(obs)

    # sanity check that no index in the dataset is >= num_obs or < 0.
    assert check_obs(num_obs, obs) == True    

    # output the mean and std_dev for this observation sequence with ID 
    (mean, std) = find_mean_std(obs)
    mean_std_file = open(os.path.join(pickle_dir, 
                                      'mean_std_' + id_pickle + '.txt'), 
                         'w+')
    mean_std_file.write("Mean: " + str(mean) + '\n' + "SD: " + str(std) + '\n')
    mean_std_file.close()  
    
    # perform training on the list of observations obs 
    (S, A, O) = baum_welch(num_states, num_obs, obs, tolerance) 

    print "FINAL TRANSITION MATRIX IS: \n", A 
    print "FINAL OBSERVATION MATRIX IS: \n", O 

    # pickle the results
    transition_file = open(os.path.join(pickle_dir, 
                                        'transition_' + id_pickle + '.npy'), 
                           'w+')
    observation_file = open(os.path.join(pickle_dir, 
                                         'observation_' + id_pickle + '.npy'), 
                            'w+')
    start_file = open(os.path.join(pickle_dir, 
                                   'start_' + id_pickle + '.npy'), 
                      'w+')

    np.save(transition_file, A)
    np.save(observation_file, O)
    np.save(start_file, S) 

    transition_file.close() 
    observation_file.close() 
    start_file.close() 

    sys.exit() 

################################################################################
# MAIN
################################################################################

if __name__ == '__main__':
    main(sys.argv)


    

