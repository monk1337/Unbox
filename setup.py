# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setup(
    name              = 'deepmap',
    version           = '0.0.1',
    description       = 'Micro Framework for Unsupervised learning tools',
    install_requires  = requirements,
    url               = 'https://github.com/monk1337/Unbox',

)