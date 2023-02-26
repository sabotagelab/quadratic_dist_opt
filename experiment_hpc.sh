#!/usr/bin/env bash
#SBATCH -J quad_dist_opt_experiment						  # name of job
#SBATCH -A eecs	  # name of my sponsored account, e.g. class or research group, NOT ONID!
#SBATCH -p dgx								  # name of partition or queue
#SBATCH -c 4
#SBATCH -o quad_dist_opt.out				  # name of output file for this submission script
#SBATCH -e quad_dist_opt.err				  # name of error file for this submission script

source /nfs/stak/users/frondan/.bashrc
conda activate quad_dist_opt


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
# python experiment_distributed.py --results_file fairness3 --N 3 --H 5 --eps_bounds 25 --notion 3 --alpha 0.0001
# python experiment_distributed.py --results_file fairness3 --N 5 --H 5 --eps_bounds 25 --notion 3 --alpha 0.0001
# python experiment_distributed.py --results_file fairness3 --N 7 --H 5 --eps_bounds 25 --notion 3 --alpha 0.0001
# python experiment_distributed.py --results_file fairness3 --N 10 --H 5 --eps_bounds 25 --notion 3 --alpha 0.0001

# python experiment_distributed.py --results_file fairness3 --N 15 --H 5 --eps_bounds 25 --Ubox 300 --notion 3 --alpha 0.0001
# python experiment_distributed.py --results_file fairness3 --N 20 --H 5 --eps_bounds 25 --Ubox 300 --notion 3 --alpha 0.0001

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


echo 'CENTRAL PROBLEMS'
# echo 'NOTION 0'
# python experiment_central.py --results_file fairness0 --N 3 --H 5 --eps_bounds 25
# python experiment_central.py --results_file fairness0 --N 5 --H 5 --eps_bounds 25
# python experiment_central.py --results_file fairness0 --N 7 --H 5 --eps_bounds 25
# python experiment_central.py --results_file fairness0 --N 10 --H 5 --eps_bounds 25

# python experiment_central.py --results_file fairness0 --N 15 --H 5 --eps_bounds 25 --Ubox 300
# python experiment_central.py --results_file fairness0 --N 20 --H 5 --eps_bounds 25 --Ubox 300


# echo 'NOTION 1'
# python experiment_central.py --results_file fairness1 --N 3 --H 5 --eps_bounds 25 --notion 1
# python experiment_central.py --results_file fairness1 --N 5 --H 5 --eps_bounds 25 --notion 1
# python experiment_central.py --results_file fairness1 --N 7 --H 5 --eps_bounds 25 --notion 1
# python experiment_central.py --results_file fairness1 --N 10 --H 5 --eps_bounds 25 --notion 1

# python experiment_central.py --results_file fairness1 --N 15 --H 5 --eps_bounds 25 --Ubox 300 --notion 1
# python experiment_central.py --results_file fairness1 --N 20 --H 5 --eps_bounds 25 --Ubox 300 --notion 1


# echo 'NOTION 2'
# python experiment_central.py --results_file fairness2 --N 3 --H 5 --eps_bounds 25 --notion 2
# python experiment_central.py --results_file fairness2 --N 5 --H 5 --eps_bounds 25 --notion 2
# python experiment_central.py --results_file fairness2 --N 7 --H 5 --eps_bounds 25 --notion 2
# python experiment_central.py --results_file fairness2 --N 10 --H 5 --eps_bounds 25 --notion 2

# python experiment_central.py --results_file fairness2 --N 15 --H 5 --eps_bounds 25 --Ubox 300 --notion 2
# python experiment_central.py --results_file fairness2 --N 20 --H 5 --eps_bounds 25 --Ubox 300 --notion 2


echo 'NOTION 3'
# python experiment_central.py --results_file fairness3 --N 3 --H 5 --eps_bounds 25 --notion 3 --alpha 0.0001
# python experiment_central.py --results_file fairness3 --N 5 --H 5 --eps_bounds 25 --notion 3 --alpha 0.0001
# python experiment_central.py --results_file fairness3 --N 7 --H 5 --eps_bounds 25 --notion 3 --alpha 0.0001
# python experiment_central.py --results_file fairness3 --N 10 --H 5 --eps_bounds 25 --notion 3 --alpha 0.0001

# python experiment_central.py --results_file fairness3 --N 15 --H 5 --eps_bounds 25 --Ubox 300 --notion 3 --alpha 0.0001
# python experiment_central.py --results_file fairness3 --N 20 --H 5 --eps_bounds 25 --Ubox 300 --notion 3 --alpha 0.0001
python experiment_central.py --results_file fairness3 --N 3 --H 5 --eps_bounds 25 --notion 3
python experiment_central.py --results_file fairness3 --N 5 --H 5 --eps_bounds 25 --notion 3
python experiment_central.py --results_file fairness3 --N 7 --H 5 --eps_bounds 25 --notion 3
python experiment_central.py --results_file fairness3 --N 10 --H 5 --eps_bounds 25 --notion 3

python experiment_central.py --results_file fairness3 --N 15 --H 5 --eps_bounds 25 --Ubox 300 --notion 3
python experiment_central.py --results_file fairness3 --N 20 --H 5 --eps_bounds 25 --Ubox 300 --notion 3


echo 'NOTION 4'
python experiment_central.py --results_file fairness4 --N 3 --H 5 --eps_bounds 25 --notion 4
python experiment_central.py --results_file fairness4 --N 5 --H 5 --eps_bounds 25 --notion 4
python experiment_central.py --results_file fairness4 --N 7 --H 5 --eps_bounds 25 --notion 4
python experiment_central.py --results_file fairness4 --N 10 --H 5 --eps_bounds 25 --notion 4

python experiment_central.py --results_file fairness4 --N 15 --H 5 --eps_bounds 25 --Ubox 300 --notion 4
python experiment_central.py --results_file fairness4 --N 20 --H 5 --eps_bounds 25 --Ubox 300 --notion 4

echo 'NOTION 5'
python experiment_central.py --results_file fairness5 --N 3 --H 5 --eps_bounds 25 --notion 5
python experiment_central.py --results_file fairness5 --N 5 --H 5 --eps_bounds 25 --notion 5
python experiment_central.py --results_file fairness5 --N 7 --H 5 --eps_bounds 25 --notion 5
python experiment_central.py --results_file fairness5 --N 10 --H 5 --eps_bounds 25 --notion 5

python experiment_central.py --results_file fairness5 --N 15 --H 5 --eps_bounds 25 --Ubox 300 --notion 5
python experiment_central.py --results_file fairness5 --N 20 --H 5 --eps_bounds 25 --Ubox 300 --notion 5
