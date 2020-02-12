# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from setuptools import setup, find_packages
import re

with open('README.rst', 'rb') as f:
    install = f.read().decode('utf-8')

with open('CHANGELOG.rst', 'rb') as f:
    changelog = f.read().decode('utf-8')

classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Environment :: Console',
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Natural Language :: English',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3']

with open('.travis.yml', 'rb') as f:
    lines = f.read().decode('utf-8')
    for version in re.findall('python: 3.[0-9]', lines):
        classifiers.append('Programming Language :: Python :: 3.%s' % version[-1])

long_description = '\n\n'.join((install, changelog))

setup(
    name='pandapower',
    version='2.2.1',
    author='Leon Thurner, Alexander Scheidler',
    author_email='leon.thurner@uni-kassel.de, alexander.scheidler@iee.fraunhofer.de',
    description='Convenient Power System Modelling and Analysis based on PYPOWER and pandas',
    long_description=long_description,
    url='http://www.pandapower.org',
    license='BSD',
    install_requires=["pandas>=0.17, <1",
                      "networkx",
                      "scipy",
                      "numpy>=0.11",
                      "packaging"],
    extras_require={":python_version<'3.0'": ["future"],
                    "docs": ["numpydoc", "sphinx", "sphinx_rtd_theme"],
                    "plotting": ["plotly", "matplotlib", "python-igraph"]},
    packages=find_packages(),
    include_package_data=True,
    classifiers=classifiers
)
