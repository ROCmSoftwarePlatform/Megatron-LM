#!/bin/bash

set -x
python -m pytest --color=yes --csv output/test_report.csv pytest_sample.py 
