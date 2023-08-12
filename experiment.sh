#!/usr/bin/env bash

echo 'DISTRIBUTED PROBLEMS'
echo 'NOTION 0'
python experiment_distributed.py --N 3 --H 5 --eps_bounds 25 --sd 0.5
python experiment_distributed.py --N 5 --H 5 --eps_bounds 25 --sd 0.5
python experiment_distributed.py --N 7 --H 5 --eps_bounds 25 --sd 0.5
python experiment_distributed.py --N 10 --H 5 --eps_bounds 25 --sd 0.5

python experiment_distributed.py --N 15 --H 5 --eps_bounds 25 --Ubox 300 --sd 0.5
python experiment_distributed.py --N 20 --H 5 --eps_bounds 25 --Ubox 300 --sd 0.5


echo 'NOTION 1'
python experiment_distributed.py --N 3 --H 5 --eps_bounds 25 --notion 1 --sd 0.5
python experiment_distributed.py --N 5 --H 5 --eps_bounds 25 --notion 1 --sd 0.5
python experiment_distributed.py --N 7 --H 5 --eps_bounds 25 --notion 1 --sd 0.5
python experiment_distributed.py --N 10 --H 5 --eps_bounds 25 --notion 1 --sd 0.5

python experiment_distributed.py --N 15 --H 5 --eps_bounds 25 --Ubox 300 --notion 1 --sd 0.5
python experiment_distributed.py --N 20 --H 5 --eps_bounds 25 --Ubox 300 --notion 1 --sd 0.5


echo 'NOTION 2'
python experiment_distributed.py --N 3 --H 5 --eps_bounds 25 --notion 2 --sd 0.5
python experiment_distributed.py --N 5 --H 5 --eps_bounds 25 --notion 2 --sd 0.5
python experiment_distributed.py --N 7 --H 5 --eps_bounds 25 --notion 2 --sd 0.5
python experiment_distributed.py --N 10 --H 5 --eps_bounds 25 --notion 2 --sd 0.5

python experiment_distributed.py --N 15 --H 5 --eps_bounds 25 --Ubox 300 --notion 2 --sd 0.5
python experiment_distributed.py --N 20 --H 5 --eps_bounds 25 --Ubox 300 --notion 2 --sd 0.5


echo 'NOTION 3'
python experiment_distributed.py --N 3 --H 5 --eps_bounds 25 --notion 3 --sd 0.5
python experiment_distributed.py --N 5 --H 5 --eps_bounds 25 --notion 3 --sd 0.5
python experiment_distributed.py --N 7 --H 5 --eps_bounds 25 --notion 3 --sd 0.5
python experiment_distributed.py --N 10 --H 5 --eps_bounds 25 --notion 3 --sd 0.5

python experiment_distributed.py --N 15 --H 5 --eps_bounds 25 --Ubox 300 --notion 3 --sd 0.5
python experiment_distributed.py --N 20 --H 5 --eps_bounds 25 --Ubox 300 --notion 3 --sd 0.5


echo 'NOTION 4'
python experiment_distributed.py --N 3 --H 5 --eps_bounds 25 --notion 4 --sd 0.5
python experiment_distributed.py --N 5 --H 5 --eps_bounds 25 --notion 4 --sd 0.5
python experiment_distributed.py --N 7 --H 5 --eps_bounds 25 --notion 4 --sd 0.5
python experiment_distributed.py --N 10 --H 5 --eps_bounds 25 --notion 4 --sd 0.5

python experiment_distributed.py --N 15 --H 5 --eps_bounds 25 --Ubox 300 --notion 4 --sd 0.5
python experiment_distributed.py --N 20 --H 5 --eps_bounds 25 --Ubox 300 --notion 4 --sd 0.5


echo 'NOTION 5'
python experiment_distributed.py --N 3 --H 5 --eps_bounds 25 --notion 5 --sd 0.5
python experiment_distributed.py --N 5 --H 5 --eps_bounds 25 --notion 5 --sd 0.5
python experiment_distributed.py --N 7 --H 5 --eps_bounds 25 --notion 5 --sd 0.5
python experiment_distributed.py --N 10 --H 5 --eps_bounds 25 --notion 5 --sd 0.5

python experiment_distributed.py --N 15 --H 5 --eps_bounds 25 --Ubox 300 --notion 5 --sd 0.5
python experiment_distributed.py --N 20 --H 5 --eps_bounds 25 --Ubox 300 --notion 5 --sd 0.5


echo 'CENTRALIZED PROBLEMS'
echo 'NOTION 0'
python experiment_central.py --N 3 --H 5 --eps_bounds 25 --sd 0.5
python experiment_central.py --N 5 --H 5 --eps_bounds 25 --sd 0.5
python experiment_central.py --N 7 --H 5 --eps_bounds 25 --sd 0.5
python experiment_central.py --N 10 --H 5 --eps_bounds 25 --sd 0.5

python experiment_central.py --N 15 --H 5 --eps_bounds 25 --Ubox 300 --sd 0.5
python experiment_central.py --N 20 --H 5 --eps_bounds 25 --Ubox 300 --sd 0.5


echo 'NOTION 1'
python experiment_central.py --N 3 --H 5 --eps_bounds 25 --notion 1 --sd 0.5
python experiment_central.py --N 5 --H 5 --eps_bounds 25 --notion 1 --sd 0.5
python experiment_central.py --N 7 --H 5 --eps_bounds 25 --notion 1 --sd 0.5
python experiment_central.py --N 10 --H 5 --eps_bounds 25 --notion 1 --sd 0.5

python experiment_central.py --N 15 --H 5 --eps_bounds 25 --Ubox 300 --notion 1 --sd 0.5
python experiment_central.py --N 20 --H 5 --eps_bounds 25 --Ubox 300 --notion 1 --sd 0.5


echo 'NOTION 2'
python experiment_central.py --N 3 --H 5 --eps_bounds 25 --notion 2 --sd 0.5
python experiment_central.py --N 5 --H 5 --eps_bounds 25 --notion 2 --sd 0.5
python experiment_central.py --N 7 --H 5 --eps_bounds 25 --notion 2 --sd 0.5
python experiment_central.py --N 10 --H 5 --eps_bounds 25 --notion 2 --sd 0.5

python experiment_central.py --N 15 --H 5 --eps_bounds 25 --Ubox 300 --notion 2 --sd 0.5
python experiment_central.py --N 20 --H 5 --eps_bounds 25 --Ubox 300 --notion 2 --sd 0.5


echo 'NOTION 3'
python experiment_central.py --N 3 --H 5 --eps_bounds 25 --notion 3 --sd 0.5
python experiment_central.py --N 5 --H 5 --eps_bounds 25 --notion 3 --sd 0.5
python experiment_central.py --N 7 --H 5 --eps_bounds 25 --notion 3 --sd 0.5
python experiment_central.py --N 10 --H 5 --eps_bounds 25 --notion 3 --sd 0.5

python experiment_central.py --N 15 --H 5 --eps_bounds 25 --Ubox 300 --notion 3 --sd 0.5
python experiment_central.py --N 20 --H 5 --eps_bounds 25 --Ubox 300 --notion 3 --sd 0.5


echo 'NOTION 4'
python experiment_central.py --N 3 --H 5 --eps_bounds 25 --notion 4 --sd 0.5
python experiment_central.py --N 5 --H 5 --eps_bounds 25 --notion 4 --sd 0.5
python experiment_central.py --N 7 --H 5 --eps_bounds 25 --notion 4 --sd 0.5
python experiment_central.py --N 10 --H 5 --eps_bounds 25 --notion 4 --sd 0.5

python experiment_central.py --N 15 --H 5 --eps_bounds 25 --Ubox 300 --notion 4 --sd 0.5
python experiment_central.py --N 20 --H 5 --eps_bounds 25 --Ubox 300 --notion 4 --sd 0.5


echo 'NOTION 5'
python experiment_central.py --N 3 --H 5 --eps_bounds 25 --notion 5 --sd 0.5
python experiment_central.py --N 5 --H 5 --eps_bounds 25 --notion 5 --sd 0.5
python experiment_central.py --N 7 --H 5 --eps_bounds 25 --notion 5 --sd 0.5
python experiment_central.py --N 10 --H 5 --eps_bounds 25 --notion 5 --sd 0.5

python experiment_central.py --N 15 --H 5 --eps_bounds 25 --Ubox 300 --notion 5 --sd 0.5
python experiment_central.py --N 20 --H 5 --eps_bounds 25 --Ubox 300 --notion 5 --sd 0.5
