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
generating poems. The contributors to the original project at Caltech were 
Dryden Bouamalay, Ritwik Anand, and Audrey Huang.

The project uses a context-free grammar (via PLY) for parsing the poems to 
create training data. This training data is then fed to the Baum-Welch algorithm
to train matrices. Finally, these matrices are used to generate a sample poem.

## Installation

Install the module with:

```
$ python setup.py install
```

For PyEnchant to function correctly it may be necessary to install `aspell-en`
using your distribution's package manager.

## Run

After installing the module, the `hmm` script should be available:

```
$ hmm -s <states> -t <tolerance>
```

where `<states>` is the number of hidden states you wish to use for the model,
and `<tolerance>` is the delta at which training will terminate. For example,
the following should completely quickly:

```
$ hmm -s 10 -t 0.01 
```

I appreciate any feedback. Thanks!