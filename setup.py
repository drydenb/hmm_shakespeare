#!/usr/bin/env python
# -*- coding: UTF-8 -*-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Generate poetry using Hidden Markov Models',
    'author': 'Dryden Bouamalay',
    'url': 'https://github.com/drydenb/hmm_shakespeare',
    'author_email': 'bouamalayd@gmail.com',
    'version': '1.0',
    'install_requires': [
        'nose',
        'appdirs==1.4.3',
        'numpy==1.11.2',
        'packaging==16.8',
        'ply==3.9',
        'pyenchant==1.6.8',
        'pyparsing==2.2.0',
        'scipy==0.18.1',
        'six==1.10.0',
    ],
    'packages': ['hmm'],
    'scripts': ['bin/hmm'],
    'name': 'hmm',
    'package_data': {'hmm': ['resources/*.txt']}
}

setup(**config)