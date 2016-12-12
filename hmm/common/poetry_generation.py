
# coding: utf-8

# In[50]:

from __future__ import print_function
from __future__ import division
from scipy.integrate import quad
from scipy.linalg import norm
import random 
import os 
import numpy as np 
import pickle



def convert_syllables_dict(s_dict):

    # generate new dictionary that maps words to 
    # number of syllables (int) 
    new_dict = {} 
    for k, v in s_dict.iteritems():
        if type(v) is list: 
            new_dict[k] = v[0] 
        if type(v) is int: 
            new_dict[k] = v 

    # check types 
    for k, v in new_dict.iteritems():
        assert type(new_dict[k]) is int 
    return new_dict 

# def check_syllables_dict(s_dict):
#     for k, v in s_dict.iteritems():
#         if type(v) is list:
#             if len(v) != 1:
#                 print(k)
#                 print(v)
#         else:
#             assert type(v) is int 
#     return True 

def convert_to_words(n):
    words = []
    for num in n:
        words.append(states_to_words_dict[num])
        #words.append("niggerdly")
    
    return words

def check_syllables(words):
    syllables = 0
    for word in words:
        # print(type(syllables_dict[word]))
        # print(word)
        # print(syllables_dict['wherefore'])
        # print(type(syllables_dict[word]))
        syllables += syllables_dict[word]
    
    return syllables

def pick_start(S):
    
    pick = random.random()
    
    for i in range(S.shape[0]):
        pick = pick - S[i]
        if pick <= 0:
            return i
    
    
    return 0

def pick_rhyming_start(S, A, O, rhyme):
    
    rhyme_start = np.zeros(S.shape)
    for i in range(O.shape[0]):
        for j in range(rhyme_start.shape[0]):
            rhyme_start[j] += O[i][rhyme]*A[j][i]
    
    return rhyme_start

def pick_next(A, current):
    
    pick = random.random()
    for i in range(A.shape[0]):
        try:
            pick = pick - A[i][current]
        except IndexError:
            print(type(i))
            print(type(current))
        if pick <= 0:
            return i
    
    
    return 0

def convert_to_observed(states, O):
    observed = []
    
    for state in states:
        pick = random.random()
        for i in range(O.shape[1]):
            pick = pick - O[state][i]
            if pick <= 0:
                observed.append(i)
                break
    
    
    return observed

def gen_line(A, O, start, length, offset):
    
    syllables_right = False
    
    while (syllables_right == False):
        
        hidden_states = []
        current = start
        hidden_states.append(current)
        
        for i in range(int(length - 1)):
            current = pick_next(A, current)
            hidden_states.append(current)
        
        print (hidden_states)
        observed_states = convert_to_observed(hidden_states, O)
        print (observed_states)
        current_line = convert_to_words(observed_states)
        
        # syllables_right = True
        syllables = check_syllables(current_line)
        
        if syllables == (10 - offset):
            syllables_right = True
            
    return current_line[::-1]  

def gen_couplets(A, O, S, avg_words, std):
    length = int(np.random.normal(avg_words, std))
    print (length)
    start = pick_start(S)

    rhyme1 = random.choice(rhyming_dict.keys()) 
    rhyme2 = rhyming_dict[rhyme1]

    line1 = gen_line(A, O, start, length - 1, syllables_dict[rhyme1]) + [rhyme1]

    # length = int(np.random.normal(avg_words, std))
    # print (length)
    # start = pick_start(S)
    # line2 = gen_line(A, O, start, length)
    
    # length = np.random.normal(avg_words, std) - 1
    # while True:

    #     print("in while")
    #     try:
    #         rhyming = random.choice(rhyming_dict[line1[-1]])
    #     except KeyError: 
    #         rhyming = random.choice(rhyming_dict[random.choice(rhyming_dict.keys())])
    #     if rhyming in words_to_state_dict.keys():
    #         break

   

    # S_rhyme = pick_rhyming_start(line1[-1])
    # if rhyming not in words_to_state_dict.keys(): 
    # start = pick_rhyming_start(S, A, O, words_to_state_dict[rhyming])
    line2 = gen_line(A, O, pick_start(S), length, syllables_dict[rhyme2]) + [rhyme2]

    print(rhyme1, rhyme2)
    print (line1[-1], line2[-1])
    print (line1, line2)
    
    return (line1, line2)

def poem_gen(S, A, O, avg_words, std):
    
    poem = []
    for i in range(14):
        poem.append([])
    
    for i in range(3):
        for j in range(2):
            (poem[4*i + j], poem[4*i + j + 2]) = gen_couplets(A, O, S, avg_words, std)
        
    (poem[12], poem[13]) = gen_couplets(A, O, S, avg_words, std)

    return poem
    
    
if __name__ == '__main__':


    # L = 30
    # M = 10

    # S = np.random.uniform(size=L)    # initialize a start state distribution S for the HMM 
    # S = np.divide(S, np.sum(S))      # normalize the vector to 1 

    # the rows of A are the target states and the columns of A are the start states. 
        # given a start state, one of the target states must be choosen so each column is normalized
    # A = np.random.rand(L, L) 
    # for i in range(L): 
    #     A[:,i] = np.divide(A[:,i], np.sum(A[:,i]))    

    #     # given some hidden state, there must be some observation, so every row of this matrix should
    #     # be normalized
    # O = np.random.rand(L, M) 
    # for i in range(L):
    #     O[i,:] = np.divide(O[i,:], np.sum(O[i,:])) 

    average_length = 8.15977443609
    std = 1.1474220639
    
    A = np.load(os.getcwd() + '/pickles/full_50_states_001_toler/transition_full.npy', 'r') 
    O = np.load(os.getcwd() + '/pickles/full_50_states_001_toler/observation_full.npy', 'r') 
    S = np.load(os.getcwd() + '/pickles/full_50_states_001_toler/start_full.npy', 'r') 

    A = np.transpose(A) 

    # A = pickle.load( open( "transition.npy", "rb" ) )
    # O = pickle.load( open( "observation.npy", "rb" ) )

    #S = pickle.load( open( "save.p", "rb" ) )
    #rhyming_dict = pickle.load( open( "rhyme_dic.p", "rb" ) )
    nonhomogenous_syllables_dict = pickle.load( open( "./pickles/syl_dic.p", "rb" ) )
    syllables_dict = convert_syllables_dict(nonhomogenous_syllables_dict)
    states_to_words_dict = pickle.load( open( "./pickles/index_to_word.p", "rb" ) )
    words_to_state_dict = {v: k for k, v in states_to_words_dict.iteritems()}
    rhyming_dict = pickle.load( open( "./pickles/rhyme_dic.p", "rb" ) )
    # print(states_to_words_dict)
    # check_syllables_dict(syllables_dict)
    

    poem = (poem_gen(S, A, O, average_length, std))
    for line in poem:
        print (line)
    # print(rhyming_dict)
# In[ ]:



