from __future__ import print_function
from __future__ import division
from scipy.integrate import quad
import random 
import numpy as np 
import codecs
import string
import re
import cPickle as cp
from nltk import pos_tag
from nltk import word_tokenize
import collections

from nltk.corpus import cmudict

d = cmudict.dict() # dicionary of syllables from cmudict

def check_unique(unique): 
	out = open('unique.txt', 'w')

	for i in unique: 
		out.write(i + '\n')
	out.close()

# parses the text file 'shakespeare.txt' and adds each unique word to a dictionary,
# WORD_DIC, with a unique index 
def parse(word_dic, index_dic): 

	# open 'shakespeare.txt'
	inp = open('shakespeare.txt', 'r')

	# use regex to parse unique words into list 'unique'
	text = inp.read().lower() # make everything lowercase
	text = re.sub('[^a-z\ \'\-]+', ' ', text) # replace punctuation with spaces
	text = re.sub("(?=.*\w)^(\w|')+", ' ', text)

	words = list(text.split()) # split file into a list of words by spaces
	unique = list(set(words)) # get unique list of words
	unique.remove("'")

	counter = collections.Counter(words)
	cp.dump(counter, open('./pickles/count_dic.p', 'wb'))
	print(len(count_dic))

	print("number unique words: ", len(unique))

	# for each unique word, add as key to dictionary with unique index as value
	i = 0
	for word in unique: 
		word_dic[word] = i
		i += 1

	index_dic = {y:x for x,y in word_dic.iteritems()}
	cp.dump(word_dic, open('./pickles/word_to_index.p', 'wb'))
	cp.dump(index_dic, open('./pickles/index_to_word.p', 'wb'))

	return word_dic, index_dic, unique, counter


def check(word_list): 
	out = open('./pickles/check.txt', 'w')

	for i in word_list: 
		for j in i: 
			out.write(index_dic[j] + ' ')
		out.write('\n')

def check_pos(pos_list): 
	out = open('check_pos.txt', 'w')

	for i in pos_list: 
		for j in i: 
			out.write(j + ' ')
		out.write('\n')


		 

def get_list(index_dic): 
	inp = open('shakespeare.txt', 'r')

	word_list = []
	line_list = []

	for line in inp: 
		line = line.lower()
		line = re.sub('[^a-z\ \'\-]+', ' ', line) # replace punctuation with spaces
		words = list(line.split(' '))
		words = filter(None, words)
		if "'" in words: 
			words.remove("'")
		# print(words, len(words))

		if len(words) > 1: 
			line_list.append(len(words))



	inp = open('shakespeare.txt', 'r')

	text = inp.read().lower() # make everything lowercase
	text = re.sub('[^a-z\ \'\-]+', ' ', text) # replace punctuation with spaces
	text = re.sub("(?=.*\w)^(\w|')+", '', text)

	words = list(text.split()) # split file into a list of words by spaces
	words = filter(lambda a: a != "'", words)

	# generate index for each line 
	ind = 0
	for num in line_list: 
		indices = []
		for i in range(num): 
			indices.append(word_dic[words[ind]])
			
			ind += 1
		
		word_list.append(indices[::-1])
	check(word_list)

	cp.dump(word_list, open('./pickles/word_list.p', 'wb'))
	cp.dump(word_list, open('./pickles/sonnet_to_index.p', 'wb'))

	return word_list, line_list

def pos(line_list): 
	tag_dic = {}
	pos_dic = {}
	tag_list = []
	inp = open('shakespeare.txt', 'r')

	text = inp.read().lower() # make everything lowercase
	text = re.sub('[^a-z\ \'\-]+', ' ', text) # replace punctuation with spaces
	text = re.sub("(?=.*\w)^(\w|')+", ' ', text)

	# text = word_tokenize(text)
	# tagged = pos_tag(text)

	words = text.split()
	tagged = pos_tag(words)


	# generate tags for each line
	ind = 0
	for num in line_list: 
		tags = []
		for i in range(num): 
			tags.append(tagged[ind][1])
			ind += 1
		tag_list.append(tags)
	check_pos(tag_list)

	
	# generate dictionary of tag : words
	for tag in tagged: 
		if tag[1] in tag_dic: 
			tag_dic[tag[1]].append(tag[0])
		if tag[0] in pos_dic: 
			pos_dic[tag[0]].append(tag[1])
		if tag[1] not in tag_dic: 
			tag_dic[tag[1]] = [tag[0]]
		if tag[0] not in pos_dic: 
			pos_dic[tag[0]] = [tag[1]]


	# save the tag dictionary 
	cp.dump(tag_dic, open('./pickles/pos_to_words.p', 'wb'))
	cp.dump(pos_dic, open('./pickles/words_to_pos.p', 'wb'))

	return tag_dic, tag_list, pos_dic


def pos_prob(tag_list): 
	for line in tag_list: 
		for word in line: 
			print('')

def rhyme(rhyme_dic): 
	inp = open('shakespeare.txt', 'r')
	count = 1
	poem = []

	for line in inp: 
		
		line = line.lower() 
		line = re.sub('[^a-z\ \'\-]+', ' ', line)
		line = re.sub("(?=.*\w)^(\w|')+", ' ', line)
		words = list(line.split()) 

		if len(words) > 1: 
			poem.append(words[-1])

			count += 1

		if count > 14: 
			for word in poem: 
				if word not in rhyme_dic: 
					rhyme_dic[word] = []

			# first quatrain
			rhyme_dic[poem[0]] = poem[2]
			rhyme_dic[poem[2]] = poem[0]

			rhyme_dic[poem[1]] = poem[3]
			rhyme_dic[poem[3]] = poem[1]

			# second quatrain
			rhyme_dic[poem[4]] = poem[6]
			rhyme_dic[poem[6]] = poem[4]

			rhyme_dic[poem[5]] = poem[7]
			rhyme_dic[poem[7]] = poem[5]

			# third quatrain
			rhyme_dic[poem[8]] = poem[10]
			rhyme_dic[poem[10]] = poem[8]

			rhyme_dic[poem[11]] = poem[9]
			rhyme_dic[poem[9]] = poem[11]

			# couplet
			rhyme_dic[poem[12]] = poem[13]
			rhyme_dic[poem[13]] = poem[12]


			poem = []
			count = 1

	cp.dump(rhyme_dic, open('./pickles/rhyme_dic.p', 'wb'))
	return rhyme_dic



# there are a bunch of apostrophed, dashed, weirdly spelled, or archaic words that
# aren't in the cmu dictionary. we will have to manually determine the number of 
# syllables. 
# 
# though there are likely some errors, generally, we can state that the
# number of syllables in a word is equal to the number of vowels. however we
# need to take into account exceptions, for example diphthongs which are groups
# of vowels that only count as one syllable, or words ending in 'sm' which is
# two syllables, not one. all of these exceptions are listed below. 
# 
# stores word as key and number of syllables as value in "BAD_DICT"
def bad_syllables(bad_dic, not_in_dic): 

	# list of diphthongs, which all count as one syllable. i didn't include:
	# 'ia' as in 'variation'
	# 'io' as in 'violet', but i check for '-tion' later
	# 'ui' as in 'ruin'
	# 'oe' as in 'goest'
	thongs = ['aa', 'ae', 'ai', 'ao', 'au', 'ay', 'ea', 'ee', 'ei', 'eo', 'eu', 'ey']
	thongs += ['ii', 'ie', 'iu', 'oa', 'oi', 'oo', 'ou', 'oy', 'ua', 'ue', 'uo', 'uy']
	
	# list of past tense versions of diphthongs, which count as one syllable. didn't include: 
	# 'oed' as in 'forgoed'
	past_thongs = ['aed', 'eed', 'ied', 'ued']
	vowels = 'aeiouy'

	for i in not_in_dic: 
		# initiate number of syllables to zero
		num_syl = 0

		# if first word in hyphenated phrase ends with e, first subtract a syllable
		if 'e-' in i:  
			num_syl -= 1

		# remove all dashes and apostrophes
		word = i.translate(string.maketrans("",""), string.punctuation)
		
		# count number of vowels
		for char in word: 
			if char in vowels: 
				num_syl += 1

		# subtract number of diphthongs 
		for thong in thongs: 
			if thong in word: 
				num_syl -= 1

		# if ends in 'e', subtract one vowel/syllable
		if word[-1] == 'e' and word[-2] not in ['l', 'r']: 
			num_syl -= 1

		# if ends in 'ed' or 'es', subtract one vowel/syllable 
		# didn't include 'es'
		if word[-2:] == 'ed' or word[-3:] == 'eds': 
			num_syl -= 1

		# the exception to the above rule is if it ends in 'ded' or something 
		if word[-3:] in ['bed', 'ded', 'ted', 'bred', 'dred', 'tred']: 
			num_syl += 1

		# if ends in 'tion', subtract one vowel/syllable since we didn't
		# count 'io' as a diphthong
		if word[-4:] == 'tion' or word[-5:] == 'tions': 
			num_syl -= 1

		# if ends in 'oing' as in 'doing', add one vowel/syllable 
		if word[-4:] == 'oing' or word[-5:] == 'oings': 
			num_syl += 1

		# if y is a consonant (when it's the first letter), subtract a vowel
		if word[0] == 'y': 
			num_syl -= 1

		# if 'uie' is in the word, ie 'quiet', there should be two syllables 
		if 'uie' in word: 
			num_syl += 1

		# if ends in 'sm', that's two syllables not one
		if word[-2:] == 'sm': 
			num_syl += 1

		# if the word ends in 'ed' but also has a diphthong, we will undercount
		# based on the rules above. so we add a syllable to our count if 
		# there is a past-tense diphthong in our word
		for past_thong in past_thongs: 
			if past_thong in word: 
				num_syl += 1

		# there are a few words that are just a consonant and a dash, but this
		# this counts as one syllable even though there are no vowels
		if num_syl == 0:
			num_syl += 1


		# the number of syllables is equal to the number of vowels
		bad_dic[i] = [num_syl]

	return bad_dic


# fills in SYL_DICT and BAD_DICT with word as key and number of syllables as value
def syllables(word_dic, syl_dic, bad_dic): 
	not_in_dic = [] # list of words not in cmudict, so we have to manually do them
	
	# go through each unique word in our dictionary of indexed words, WORD_DIC
	for key in word_dic:
		# if word is in cmudict, look up the number of syllables and store in SYL_DIC
		if key in d: 
			syl_dic[key] = list(set(len([y for y in x if y[-1].isdigit()]) for x in d[key]))
		# if it's not in cmudict, append to the list of words not in cmudict
		else: 
			not_in_dic.append(key)

	
	# call function to calculate number of syllables for each word not in cmudict
	bad_dic = bad_syllables(bad_dic, not_in_dic)

	z = syl_dic.copy() 
	z.update(bad_dic)

	cp.dump(z, open('./pickles/syl_dic.p', 'wb'))

	return syl_dic, bad_dic, z

def split_sonnet(word_list): 
	count = 1
	quatrains = []
	volta = []
	couplet = []

	for row in word_list: 
		if count <= 8: 
			quatrains.append(row)
		elif count <= 12: 
			volta.append(row)
		else: 
			couplet.append(row)
		count += 1

		if count > 14: 
			count = 1


	cp.dump(quatrains, open('./pickles/quatrains.p', 'wb'))
	cp.dump(volta, open('./pickles/volta.p', 'wb'))
	cp.dump(couplet, open('./pickles/couplet.p', 'wb'))
	return quatrains, volta, couplet



if __name__ == '__main__':
	word_dic = {} # dictionary of unique word : unique index 
	index_dic = {} # dictionary of unique index : unique word
	syl_dic = {} # dictionary of unique word : number of syllables if they're in cmudict
	bad_dic = {} # dictionary of unique word : number of syllables if they're not in cmudict
	tag_dic = {} # dictionary of parts of speech : words 
	rhyme_dic = {} # dictionary of word : rhyming words 

	
	word_dic, index_dic, unique, count_dic = parse(word_dic, index_dic) # parse file into word_dic, word : index
	
	word_list, line_list = get_list(index_dic)
	quatrains, volta, couplet = split_sonnet(word_list)
	
	tag_dic, tag_list, pos_dic = pos(line_list)

	rhyme_dic = rhyme(rhyme_dic)

	part_dic, bad_dic, syl_dic = syllables(word_dic, syl_dic, bad_dic) # parse words into syl_dic and bad_dic, word : number of syllables

