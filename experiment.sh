#!/usr/bin/env bash
#SBATCH -t 3-12:30:00
#SBATCH -J fair_uam						  # name of job
#SBATCH -A hlab	  # name of my sponsored account, e.g. class or research group, NOT ONID!
#SBATCH -p athena								  # name of partition or queue
#SBATCH -o fairfly.out				  # name of output file for this submission script
#SBATCH -e fairfly.err				  # name of error file for this submission script
#SBATCH -c 28
#SBATCH --mem=75G

source /nfs/stak/users/frondan/.bashrc
conda activate quad_dist_opt

# central
python mpc_test.py --exp_dir /3_central_fair_notion0 --trials 5 --notion 0 --N 3 --exp_params IROS2024_experiments/manhattan_params2.yaml
python mpc_test.py --exp_dir /3_central_fair_notion1 --trials 5 --notion 1 --N 3 --exp_params IROS2024_experiments/manhattan_params2.yaml
python mpc_test.py --exp_dir /3_central_fair_notion2 --trials 5 --notion 2 --N 3 --exp_params IROS2024_experiments/manhattan_params2.yaml
python mpc_test.py --exp_dir /3_central_fair_notion3 --trials 5 --notion 3 --N 3 --exp_params IROS2024_experiments/manhattan_params2.yaml
python mpc_test.py --exp_dir /3_central_fair_notion4 --trials 5 --notion 4 --N 3 --exp_params IROS2024_experiments/manhattan_params2.yaml
python mpc_test.py --exp_dir /3_central_fair_notion5 --trials 5 --notion 5 --N 3 --exp_params IROS2024_experiments/manhattan_params2.yaml


python mpc_test.py --exp_dir /5_central_fair_notion0 --trials 5 --notion 0 --N 5 --exp_params IROS2024_experiments/manhattan_params2.yaml
python mpc_test.py --exp_dir /5_central_fair_notion1 --trials 5 --notion 1 --N 5 --exp_params IROS2024_experiments/manhattan_params2.yaml
python mpc_test.py --exp_dir /5_central_fair_notion2 --trials 5 --notion 2 --N 5 --exp_params IROS2024_experiments/manhattan_params2.yaml
python mpc_test.py --exp_dir /5_central_fair_notion3 --trials 5 --notion 3 --N 5 --exp_params IROS2024_experiments/manhattan_params2.yaml
python mpc_test.py --exp_dir /5_central_fair_notion4 --trials 5 --notion 4 --N 5 --exp_params IROS2024_experiments/manhattan_params2.yaml
python mpc_test.py --exp_dir /5_central_fair_notion5 --trials 5 --notion 5 --N 5 --exp_params IROS2024_experiments/manhattan_params2.yaml


python mpc_test.py --exp_dir /7_central_fair_notion0 --trials 5 --notion 0 --N 7 --exp_params IROS2024_experiments/manhattan_params2.yaml
python mpc_test.py --exp_dir /7_central_fair_notion1 --trials 5 --notion 1 --N 7 --exp_params IROS2024_experiments/manhattan_params2.yaml
python mpc_test.py --exp_dir /7_central_fair_notion2 --trials 5 --notion 2 --N 7 --exp_params IROS2024_experiments/manhattan_params2.yaml
python mpc_test.py --exp_dir /7_central_fair_notion3 --trials 5 --notion 3 --N 7 --exp_params IROS2024_experiments/manhattan_params2.yaml
python mpc_test.py --exp_dir /7_central_fair_notion4 --trials 5 --notion 4 --N 7 --exp_params IROS2024_experiments/manhattan_params2.yaml
python mpc_test.py --exp_dir /7_central_fair_notion5 --trials 5 --notion 5 --N 7 --exp_params IROS2024_experiments/manhattan_params2.yaml


python mpc_test.py --exp_dir /10_central_fair_notion0 --trials 5 --notion 0 --N 10 --exp_params IROS2024_experiments/manhattan_params2.yaml
python mpc_test.py --exp_dir /10_central_fair_notion1 --trials 5 --notion 1 --N 10 --exp_params IROS2024_experiments/manhattan_params2.yaml
python mpc_test.py --exp_dir /10_central_fair_notion2 --trials 5 --notion 2 --N 10 --exp_params IROS2024_experiments/manhattan_params2.yaml
python mpc_test.py --exp_dir /10_central_fair_notion3 --trials 5 --notion 3 --N 10 --exp_params IROS2024_experiments/manhattan_params2.yaml
python mpc_test.py --exp_dir /10_central_fair_notion4 --trials 5 --notion 4 --N 10 --exp_params IROS2024_experiments/manhattan_params2.yaml
python mpc_test.py --exp_dir /10_central_fair_notion5 --trials 5 --notion 5 --N 10 --exp_params IROS2024_experiments/manhattan_params2.yaml


# dist
python mpc_test.py --exp_dir /3_central_fair_notion0_dist --trials 5 --notion 0 --N 3 --dist --exp_params IROS2024_experiments/manhattan_params2.yaml
python mpc_test.py --exp_dir /3_central_fair_notion0_dist --trials 5 --notion 1 --N 3 --dist --exp_params IROS2024_experiments/manhattan_params2.yaml
python mpc_test.py --exp_dir /3_central_fair_notion2_dist --trials 5 --notion 2 --N 3 --dist --exp_params IROS2024_experiments/manhattan_params2.yaml
python mpc_test.py --exp_dir /3_central_fair_notion3_dist --trials 5 --notion 3 --N 3 --dist --exp_params IROS2024_experiments/manhattan_params2.yaml
python mpc_test.py --exp_dir /3_central_fair_notion4_dist --trials 5 --notion 4 --N 3 --dist --exp_params IROS2024_experiments/manhattan_params2.yaml
python mpc_test.py --exp_dir /3_central_fair_notion5_dist --trials 5 --notion 5 --N 3 --dist --exp_params IROS2024_experiments/manhattan_params2.yaml

python mpc_test.py --exp_dir /5_central_fair_notion0_dist --trials 5 --notion 0 --N 5 --dist --exp_params IROS2024_experiments/manhattan_params2.yaml
python mpc_test.py --exp_dir /5_central_fair_notion1_dist --trials 5 --notion 1 --N 5 --dist --exp_params IROS2024_experiments/manhattan_params2.yaml
python mpc_test.py --exp_dir /5_central_fair_notion2_dist --trials 5 --notion 2 --N 5 --dist --exp_params IROS2024_experiments/manhattan_params2.yaml
python mpc_test.py --exp_dir /5_central_fair_notion3_dist --trials 5 --notion 3 --N 5 --dist --exp_params IROS2024_experiments/manhattan_params2.yaml
python mpc_test.py --exp_dir /5_central_fair_notion4_dist --trials 5 --notion 4 --N 5 --dist --exp_params IROS2024_experiments/manhattan_params2.yaml
python mpc_test.py --exp_dir /5_central_fair_notion5_dist --trials 5 --notion 5 --N 5 --dist --exp_params IROS2024_experiments/manhattan_params2.yaml

python mpc_test.py --exp_dir /7_central_fair_notion0_dist --trials 5 --notion 0 --N 7 --dist --exp_params IROS2024_experiments/manhattan_params2.yaml
python mpc_test.py --exp_dir /7_central_fair_notion1_dist --trials 5 --notion 1 --N 7 --dist --exp_params IROS2024_experiments/manhattan_params2.yaml
python mpc_test.py --exp_dir /7_central_fair_notion2_dist --trials 5 --notion 2 --N 7 --dist --exp_params IROS2024_experiments/manhattan_params2.yaml
python mpc_test.py --exp_dir /7_central_fair_notion3_dist --trials 5 --notion 3 --N 7 --dist --exp_params IROS2024_experiments/manhattan_params2.yaml
python mpc_test.py --exp_dir /7_central_fair_notion4_dist --trials 5 --notion 4 --N 7 --dist --exp_params IROS2024_experiments/manhattan_params2.yaml
python mpc_test.py --exp_dir /7_central_fair_notion5_dist --trials 5 --notion 5 --N 7 --dist --exp_params IROS2024_experiments/manhattan_params2.yaml

python mpc_test.py --exp_dir /10_central_fair_notion0_dist --trials 5 --notion 0 --N 10 --dist --exp_params IROS2024_experiments/manhattan_params2.yaml
python mpc_test.py --exp_dir /10_central_fair_notion1_dist --trials 5 --notion 1 --N 10 --dist --exp_params IROS2024_experiments/manhattan_params2.yaml
python mpc_test.py --exp_dir /10_central_fair_notion2_dist --trials 5 --notion 2 --N 10 --dist --exp_params IROS2024_experiments/manhattan_params2.yaml
python mpc_test.py --exp_dir /10_central_fair_notion3_dist --trials 5 --notion 3 --N 10 --dist --exp_params IROS2024_experiments/manhattan_params2.yaml
python mpc_test.py --exp_dir /10_central_fair_notion4_dist --trials 5 --notion 4 --N 10 --dist --exp_params IROS2024_experiments/manhattan_params2.yaml
python mpc_test.py --exp_dir /10_central_fair_notion5_dist --trials 5 --notion 5 --N 10 --dist --exp_params IROS2024_experiments/manhattan_params2.yaml
