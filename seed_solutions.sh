#!/usr/bin/env bash

python get_seed_solutions.py --N 3 --H 5 --eps_bounds 25 --notion 2 --sd 0.5 --trial 200
python get_seed_solutions.py --N 5 --H 5 --eps_bounds 25 --notion 2 --sd 0.5 --trial 200 
python get_seed_solutions.py --N 7 --H 5 --eps_bounds 25 --notion 2 --sd 0.5 --trial 200
python get_seed_solutions.py --N 10 --H 5 --eps_bounds 25 --notion 2 --sd 0.5 --trial 200

python get_seed_solutions.py --N 15 --H 5 --eps_bounds 25 --notion 2 --Ubox 300 --sd 0.5 --trial 200
python get_seed_solutions.py --N 20 --H 5 --eps_bounds 25 --notion 2 --Ubox 300 --sd 0.5 --trial 200
