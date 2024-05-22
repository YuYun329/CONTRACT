#!/usr/bin/env python

import pathlib
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

import codecs
import os.path


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(name='scContract',
      version=get_version(HERE / "contract/__init__.py"),
      packages=find_packages(),
      description='Research on deep learning-based integration of single-cell multi-omics data and inference of regulation networks',
      long_description=README,

      author='Yu Yun',
      author_email='yuy569438@gmail.com',
      url='https://github.com/YuYun329/CONTRACT',
      scripts=['CONTRACT.py'],
      install_requires=requirements,
      python_requires='>=3.8.0',
      license='MIT',

      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.8',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX :: Linux',
          'Topic :: Scientific/Engineering :: Bio-Informatics',
      ],
      )
