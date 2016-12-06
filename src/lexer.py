import sys 
import ply.lex as lex

# all possible tokens in the poems 
tokens = (
	'WORD',
	'QUOTE',
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
t_WORD      = r'\b([a-zA-Z]+)\b'
t_QUOTE     = r'\''
t_COLON     = r':'
t_COMMA     = r','
t_PERIOD    = r'\.'
t_QMARK     = r'\?'
t_HYPHEN    = r'-'
t_SEMICOLON = r';'
t_LPAREN    = r'\('
t_RPAREN    = r'\)'
t_BANG      = r'!'
t_RETURN    = r'\n'
t_ignore  = ' \t'    # ignore spaces and tabs 

# numbers require a cast to int
def t_NUMBER(t):
    r'\d+'
    t.value = int(t.value)    
    return t

# for line numbers 
# def t_newline(t):
#     r'\n+'
#     t.lexer.lineno += len(t.value)

# EOF handling rule
# def t_EOF(t):
#      return None

# for error handling 
def t_error(t):
    
    print("Illegal character '%s'" % t.value[0])
    print("Exiting...")

    # it is not acceptable to fail to parse something
    sys.exit(1)

# Build the lexer
lexer = lex.lex()

# Test it out
with open('./data/raw/shakespeare.txt', 'r') as datafile:
	data = datafile.read()

# Give the lexer some input
lexer.input(data)

# Tokenize
# while True:
#     tok = lexer.token()
#     if not tok: 
#         break      # No more input
#     print(tok)