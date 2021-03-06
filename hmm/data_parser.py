#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Developed by: Dryden Bouamalay
# Purpose: Defines a context free grammar using PLY for parsing

import ply.yacc as yacc
import sys

# get the token map from the lexer
from data_lexer import tokens


def p_poems(p):
    '''
    poems : poems poem 
    poems : poem 
    '''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1]
        p[0].append(p[2])


def p_poem(p):
    'poem : NUMBER returns lines returns'
    p[0] = p[3]


def p_lines(p):
    '''
    lines : lines line
    lines : line
    '''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1]
        p[0].append(p[2])


def p_line(p):
    'line : elements RETURN'
    p[0] = p[1]


def p_elements(p):
    '''
    elements : elements element
    elements : element
    '''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1]
        p[0].append(p[2])


def p_element(p):
    '''element : STRING
               | SINGLEQUOTE
               | COLON
               | COMMA
               | NUMBER
               | PERIOD
               | QMARK
               | HYPHEN
               | SEMICOLON
               | LPAREN
               | RPAREN
               | BANG'''
    p[0] = str(p[1])


def p_returns(p):
    '''
    returns : returns RETURN
    returns : RETURN
    '''


# Error rule for syntax errors
def p_error(p):
    print("Syntax error in input!")
    print(p)
    sys.exit(1)


# Build the parser
parser = yacc.yacc(debug=True)
