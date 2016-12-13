# Generating Poetry with Hidden Markov Models

## Purpose

This project implements from scratch the Baum-Welch algorithm to train a 
Hidden Markov Model. In particular, this project trains a HMM using Shakespearan
poetry and generates sample sonnets using the model.

This project arose out of a mini-project in machine learning for CS 155 at 
Caltech. The project has since been restructured and expanded, with all old 
source code removed. The new project has been taken up by Dryden Bouamalay. The 
contributors to the original project at Caltech were Dryden Bouamalay, Ritwik Anand, 
and Audrey Huang.

The data processing now utilizes a CFG (via PLY) for parsing training data and the 
project generally uses better coding practices throughout the codebase as a whole.  

## Dependencies

Install Python dependencies as usual in a virtual environment with 
`pip install -r requirements.txt`. In addition, for PyEnchant to function 
correctly it may be necessary to install `aspell-en` using your distribution's 
package manager. 
