#!/bin/bash
set -ex

which python3

python3 -m pip install --upgrade pip
python3 -m pip install virtualenv
python3 -m virtualenv venv

source venv/bin/activate

which python

python -m pip install -r requirements.txt
python -m pip install -r requirements_qa.txt
