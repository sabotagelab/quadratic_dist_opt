#!/usr/bin/env bash
#SBATCH -t 2-12:30:00
#SBATCH -J fair_uam						  # name of job
#SBATCH -A hlab	  # name of my sponsored account, e.g. class or research group, NOT ONID!
#SBATCH -p athena								  # name of partition or queue
#SBATCH -o fairfly.out				  # name of output file for this submission script
#SBATCH -e fairfly.err				  # name of error file for this submission script
#SBATCH -c 28
#SBATCH --mem=75G

python mpc_test.py --exp_dir /3_central_fair_notion0 --trials 200 --notion 0 --N 3
python mpc_test.py --exp_dir /3_central_fair_notion2 --trials 200 --notion 2 --N 3
python mpc_test.py --exp_dir /3_central_fair_notion0_dist --trials 200 --notion 0 --dist --N 3
python mpc_test.py --exp_dir /3_central_fair_notion2_dist --trials 200 --notion 2 --dist --N 3

python mpc_test.py --exp_dir /5_central_fair_notion0 --trials 200 --notion 0 --N 5
python mpc_test.py --exp_dir /5_central_fair_notion2 --trials 200 --notion 2 --N 5
python mpc_test.py --exp_dir /5_central_fair_notion0_dist --trials 200 --notion 0 --dist --N 5
python mpc_test.py --exp_dir /5_central_fair_notion2_dist --trials 200 --notion 2 --dist --N 5

python mpc_test.py --exp_dir /7_central_fair_notion0 --trials 200 --notion 0 --N 7
python mpc_test.py --exp_dir /7_central_fair_notion2 --trials 200 --notion 2 --N 7
python mpc_test.py --exp_dir /7_central_fair_notion0_dist --trials 200 --notion 0 --dist --N 7
python mpc_test.py --exp_dir /7_central_fair_notion2_dist --trials 200 --notion 2 --dist --N 7

python mpc_test.py --exp_dir /3_central_fair_notion0_man --trials 200 --notion 0 --N 3
python mpc_test.py --exp_dir /5_central_fair_notion0_man --trials 200 --notion 0 --N 5
