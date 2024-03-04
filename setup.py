#!/usr/bin/env python

import setuptools
import os

setuptools.setup(
    name='scikit-learn-emeralds',
    version='0.0.3',
    description='Collection of utils for scikit-learn & scipy',
    long_description="""Collection of utils for scikit-learn & scipy""",
    long_description_content_type="text/markdown",
    author='Egil Moeller, Craig W. Christensen, et al.',
    author_email='em@emeraldgeo.no',
    url='https://github.com/emerald-geomodelling/scikit-learn-emeralds',
    packages=setuptools.find_packages(),
    install_requires=[
        "seaborn",
        "pandas",
        "numpy",
        "scipy"
    ],
)
