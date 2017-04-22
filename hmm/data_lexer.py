#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Developed by: Dryden Bouamalay
# Purpose: Defines a lexer for parsing using PLY

import os
import sys
import ply.lex as lex

# all possible tokens in the poems 
tokens = (
    'STRING',
    'SINGLEQUOTE',
    'COLON',
    'COMMA',
    'NUMBER',
    'PERIOD',
    'QMARK',
    'HYPHEN',
    'SEMICOLON',
    'LPAREN',
    'RPAREN',
    'BANG',
    'RETURN',
)

# regular expression rules for tokens
t_STRING          = r'\b([a-zA-Z]+)\b'
t_SINGLEQUOTE     = r'\''
t_COLON           = r':'
t_COMMA           = r','
t_PERIOD          = r'\.'
t_QMARK           = r'\?'
t_HYPHEN          = r'-'
t_SEMICOLON       = r';'
t_LPAREN          = r'\('
t_RPAREN          = r'\)'
t_BANG            = r'!'
t_RETURN          = r'\n'
t_ignore          = ' \t'    # ignore spaces and tabs 

# numbers require a cast to int
def t_NUMBER(t):
    r'\d+'
    t.value = int(t.value)    
    return t

# for error handling 
def t_error(t):
    
    print("Illegal character '%s'" % t.value[0])
    print("Exiting...")

    # it is not acceptable to fail to parse something
    sys.exit(1)

lexer = lex.lex()

# find the data file
file_dir = os.path.dirname(os.path.realpath(__file__))
resources_dir = os.path.join(file_dir, 'resources')
shakespeare_file = os.path.join(resources_dir, 'shakespeare.txt')

with open(shakespeare_file, 'r') as datafile:
    data = datafile.read()

lexer.input(data)
