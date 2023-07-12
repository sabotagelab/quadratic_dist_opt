#!/usr/bin/env bash

echo 'DISTRIBUTED PROBLEMS'
echo 'NOTION 0'
python experiment_distributed.py --results_file fairness0 --N 3 --H 5 --eps_bounds 25
python experiment_distributed.py --results_file fairness0 --N 5 --H 5 --eps_bounds 25
python experiment_distributed.py --results_file fairness0 --N 7 --H 5 --eps_bounds 25
python experiment_distributed.py --results_file fairness0 --N 10 --H 5 --eps_bounds 25

python experiment_distributed.py --results_file fairness0 --N 15 --H 5 --eps_bounds 25 --Ubox 300
python experiment_distributed.py --results_file fairness0 --N 20 --H 5 --eps_bounds 25 --Ubox 300


echo 'NOTION 1'
python experiment_distributed.py --results_file fairness1 --N 3 --H 5 --eps_bounds 25 --notion 1
python experiment_distributed.py --results_file fairness1 --N 5 --H 5 --eps_bounds 25 --notion 1
python experiment_distributed.py --results_file fairness1 --N 7 --H 5 --eps_bounds 25 --notion 1
python experiment_distributed.py --results_file fairness1 --N 10 --H 5 --eps_bounds 25 --notion 1

python experiment_distributed.py --results_file fairness1 --N 15 --H 5 --eps_bounds 25 --Ubox 300 --notion 1
python experiment_distributed.py --results_file fairness1 --N 20 --H 5 --eps_bounds 25 --Ubox 300 --notion 1


echo 'NOTION 2'
python experiment_distributed.py --results_file fairness2 --N 3 --H 5 --eps_bounds 25 --notion 2
python experiment_distributed.py --results_file fairness2 --N 5 --H 5 --eps_bounds 25 --notion 2
python experiment_distributed.py --results_file fairness2 --N 7 --H 5 --eps_bounds 25 --notion 2
python experiment_distributed.py --results_file fairness2 --N 10 --H 5 --eps_bounds 25 --notion 2

python experiment_distributed.py --results_file fairness2 --N 15 --H 5 --eps_bounds 25 --Ubox 300 --notion 2
python experiment_distributed.py --results_file fairness2 --N 20 --H 5 --eps_bounds 25 --Ubox 300 --notion 2


echo 'NOTION 3'
python experiment_distributed.py --results_file fairness3 --N 3 --H 5 --eps_bounds 25 --notion 3
python experiment_distributed.py --results_file fairness3 --N 5 --H 5 --eps_bounds 25 --notion 3
python experiment_distributed.py --results_file fairness3 --N 7 --H 5 --eps_bounds 25 --notion 3
python experiment_distributed.py --results_file fairness3 --N 10 --H 5 --eps_bounds 25 --notion 3

python experiment_distributed.py --results_file fairness3 --N 15 --H 5 --eps_bounds 25 --Ubox 300 --notion 3
python experiment_distributed.py --results_file fairness3 --N 20 --H 5 --eps_bounds 25 --Ubox 300 --notion 3


echo 'NOTION 4'
python experiment_distributed.py --results_file fairness4 --N 3 --H 5 --eps_bounds 25 --notion 4
python experiment_distributed.py --results_file fairness4 --N 5 --H 5 --eps_bounds 25 --notion 4
python experiment_distributed.py --results_file fairness4 --N 7 --H 5 --eps_bounds 25 --notion 4
python experiment_distributed.py --results_file fairness4 --N 10 --H 5 --eps_bounds 25 --notion 4

python experiment_distributed.py --results_file fairness4 --N 15 --H 5 --eps_bounds 25 --Ubox 300 --notion 4
python experiment_distributed.py --results_file fairness4 --N 20 --H 5 --eps_bounds 25 --Ubox 300 --notion 4


echo 'NOTION 5'
python experiment_distributed.py --results_file fairness5 --N 3 --H 5 --eps_bounds 25 --notion 5
python experiment_distributed.py --results_file fairness5 --N 5 --H 5 --eps_bounds 25 --notion 5
python experiment_distributed.py --results_file fairness5 --N 7 --H 5 --eps_bounds 25 --notion 5
python experiment_distributed.py --results_file fairness5 --N 10 --H 5 --eps_bounds 25 --notion 5

python experiment_distributed.py --results_file fairness5 --N 15 --H 5 --eps_bounds 25 --Ubox 300 --notion 5
python experiment_distributed.py --results_file fairness5 --N 20 --H 5 --eps_bounds 25 --Ubox 300 --notion 5
