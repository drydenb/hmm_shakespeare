# Generating Poetry with Hidden Markov Models

## Introduction

This project implements the Baum-Welch algorithm to train a Hidden Markov Model 
(HMM) using Shakespearean poetry as a training set. In addition, the trained 
model is used to generate a sample sonnet. This is accomplished by selecting a 
seed value from the starting distribution of the model and emitting possible 
observations while propagating through hidden states.

This project arose out of a mini-project in machine learning for CS 155 at 
Caltech. The project has since been completely restructured, with all old 
source code removed. The main motivation was to create a cleaner implementation 
that utilizes a HMM without any domain specific knowledge of poetry. The result 
is an implementation that is more representative of the strength of a HMM for
generating poems. While this means that the poetry generated is mostly nonsense,
it is an interesting application. The contributors to the original project at 
Caltech were Dryden Bouamalay, Ritwik Anand, and Audrey Huang.

The project uses a context-free grammar (via PLY) for parsing the poems to 
create training data. This training data is then fed to the Baum-Welch algorithm
to train matrices. Finally, these matrices are used to generate a sample poem.

## Dependencies

Install Python dependencies using a virtual environment:

1. Configure pip, virtualenv, and virtualenvwrapper for your system:
2. Make a virtual environment for the project:
  * `mkvirtualenv hmm`
3. If the environment isn't automatically activated, activate it with:
  * `workon hmm`
4. Install all project dependencies:
  * `pip install -r requirements.txt`

In addition, for PyEnchant to function correctly it may be necessary to 
install `aspell-en` using your distribution's package manager.

## Run

After installing dependencies and activating the virtual environment, run
the program with:

  * `python src/run.py -s STATES -t TOLERANCE`

where `STATES` is the number of hidden states you wish to use for the model,
and `TOLERANCE` is the delta at which training will terminate. For example,

  * `python src/run.py -s 10 -t 0.01` 

completes relatively quickly. For more help, use `-h` or `--help`. 

As always, I appreciate any feedback. Thanks! 