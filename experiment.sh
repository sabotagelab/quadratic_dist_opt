#!/usr/bin/env bash
#SBATCH -t 3-12:30:00
#SBATCH -J fair_uam						  # name of job
#SBATCH -A hlab	  # name of my sponsored account, e.g. class or research group, NOT ONID!
#SBATCH -p athena								  # name of partition or queue
#SBATCH -o fairfly.out				  # name of output file for this submission script
#SBATCH -e fairfly.err				  # name of error file for this submission script
#SBATCH -c 28
#SBATCH --mem=75G

# source /nfs/stak/users/frondan/.bashrc
# conda activate quad_dist_opt

# python mpc_test_fixed_pos_rand_obs.py --exp_dir /exp1_star_central_no_fair --notion 2 --trials 100 --N 5
# python mpc_test_fixed_pos_rand_obs.py --exp_dir /exp1_star_central_fair0 --notion 0 --trials 100 --N 5
# python mpc_test_fixed_pos_rand_obs.py --exp_dir /exp1_star_central_fair3 --notion 3 --trials 100 --N 5
# python mpc_test_fixed_pos_rand_obs.py --exp_dir /exp1_star_central_fair4 --notion 4 --trials 100 --N 5
# python mpc_test_fixed_pos_rand_obs.py --exp_dir /exp1_star_central_fair5 --notion 5 --trials 100 --N 5

# python mpc_test_fixed_pos_rand_obs.py --exp_dir /exp1_star_dist_no_fair --notion 2 --trials 100 --N 5 --dist
# python mpc_test_fixed_pos_rand_obs.py --exp_dir /exp1_star_dist_fair0 --notion 0 --trials 100 --N 5 --dist
# python mpc_test_fixed_pos_rand_obs.py --exp_dir /exp1_star_dist_fair3 --notion 3 --trials 100 --N 5 --dist
# python mpc_test_fixed_pos_rand_obs.py --exp_dir /exp1_star_dist_fair4 --notion 4 --trials 100 --N 5 --dist
# python mpc_test_fixed_pos_rand_obs.py --exp_dir /exp1_star_dist_fair5 --notion 5 --trials 100 --N 5 --dist

# python mpc_test_fixed_pos_rand_obs.py --exp_dir /exp1_star10_central_no_fair --notion 2 --trials 100 --N 10
python mpc_test_fixed_pos_rand_obs.py --exp_dir /exp1_star10_central_fair0 --notion 0 --trials 100 --N 10
python mpc_test_fixed_pos_rand_obs.py --exp_dir /exp1_star10_central_fair3 --notion 3 --trials 100 --N 10
python mpc_test_fixed_pos_rand_obs.py --exp_dir /exp1_star10_central_fair4 --notion 4 --trials 100 --N 10
python mpc_test_fixed_pos_rand_obs.py --exp_dir /exp1_star10_central_fair5 --notion 5 --trials 100 --N 10

python mpc_test_fixed_pos_rand_obs.py --exp_dir /exp1_star10_dist_no_fair --notion 2 --trials 100 --N 10 --dist
python mpc_test_fixed_pos_rand_obs.py --exp_dir /exp1_star10_dist_fair0 --notion 0 --trials 100 --N 10 --dist
python mpc_test_fixed_pos_rand_obs.py --exp_dir /exp1_star10_dist_fair3 --notion 3 --trials 100 --N 10 --dist
python mpc_test_fixed_pos_rand_obs.py --exp_dir /exp1_star10_dist_fair4 --notion 4 --trials 100 --N 10 --dist
python mpc_test_fixed_pos_rand_obs.py --exp_dir /exp1_star10_dist_fair5 --notion 5 --trials 100 --N 10 --dist

# # central
# # python mpc_test.py --exp_dir /3_central_fair_notion0 --trials 40 --notion 0 --N 3 --exp_params IROS2024_experiments/manhattan_params2.yaml
# # python mpc_test.py --exp_dir /3_central_fair_notion2 --trials 40 --notion 2 --N 3 --exp_params IROS2024_experiments/manhattan_params2.yaml
# # python mpc_test.py --exp_dir /3_central_fair_notion4 --trials 40 --notion 4 --N 3 --exp_params IROS2024_experiments/manhattan_params2.yaml
# python mpc_test.py --exp_dir /3_central_fair_notion5 --trials 40 --notion 5 --N 3 --exp_params IROS2024_experiments/manhattan_params2.yaml


# # python mpc_test.py --exp_dir /5_central_fair_notion0 --trials 40 --notion 0 --N 5 --exp_params IROS2024_experiments/manhattan_params2.yaml
# # python mpc_test.py --exp_dir /5_central_fair_notion2 --trials 40 --notion 2 --N 5 --exp_params IROS2024_experiments/manhattan_params2.yaml
# # python mpc_test.py --exp_dir /5_central_fair_notion4 --trials 40 --notion 4 --N 5 --exp_params IROS2024_experiments/manhattan_params2.yaml
# python mpc_test.py --exp_dir /5_central_fair_notion5 --trials 40 --notion 5 --N 5 --exp_params IROS2024_experiments/manhattan_params2.yaml


# # python mpc_test.py --exp_dir /7_central_fair_notion0 --trials 40 --notion 0 --N 7 --exp_params IROS2024_experiments/manhattan_params2.yaml
# # python mpc_test.py --exp_dir /7_central_fair_notion2 --trials 40 --notion 2 --N 7 --exp_params IROS2024_experiments/manhattan_params2.yaml
# # python mpc_test.py --exp_dir /7_central_fair_notion4 --trials 40 --notion 4 --N 7 --exp_params IROS2024_experiments/manhattan_params2.yaml
# python mpc_test.py --exp_dir /7_central_fair_notion5 --trials 40 --notion 5 --N 7 --exp_params IROS2024_experiments/manhattan_params2.yaml


# # dist
# # python mpc_test.py --exp_dir /3_central_fair_notion0_dist --trials 20 --notion 0 --N 3 --dist --exp_params IROS2024_experiments/manhattan_params2.yaml
# # python mpc_test.py --exp_dir /3_central_fair_notion2_dist --trials 20 --notion 2 --N 3 --dist --exp_params IROS2024_experiments/manhattan_params2.yaml
# # python mpc_test.py --exp_dir /3_central_fair_notion4_dist --trials 20 --notion 4 --N 3 --dist --exp_params IROS2024_experiments/manhattan_params2.yaml
# # python mpc_test.py --exp_dir /3_central_fair_notion5_dist --trials 20 --notion 5 --N 3 --dist --exp_params IROS2024_experiments/manhattan_params2.yaml

# # python mpc_test.py --exp_dir /5_central_fair_notion0_dist --trials 20 --notion 0 --N 5 --dist --exp_params IROS2024_experiments/manhattan_params2.yaml
# # python mpc_test.py --exp_dir /5_central_fair_notion2_dist --trials 20 --notion 2 --N 5 --dist --exp_params IROS2024_experiments/manhattan_params2.yaml
# # python mpc_test.py --exp_dir /5_central_fair_notion4_dist --trials 20 --notion 4 --N 5 --dist --exp_params IROS2024_experiments/manhattan_params2.yaml
# # python mpc_test.py --exp_dir /5_central_fair_notion5_dist --trials 20 --notion 5 --N 5 --dist --exp_params IROS2024_experiments/manhattan_params2.yaml

# # python mpc_test.py --exp_dir /7_central_fair_notion0_dist --trials 20 --notion 0 --N 7 --dist --exp_params IROS2024_experiments/manhattan_params2.yaml
# # python mpc_test.py --exp_dir /7_central_fair_notion2_dist --trials 20 --notion 2 --N 7 --dist --exp_params IROS2024_experiments/manhattan_params2.yaml
# # python mpc_test.py --exp_dir /7_central_fair_notion4_dist --trials 20 --notion 4 --N 7 --dist --exp_params IROS2024_experiments/manhattan_params2.yaml
# # python mpc_test.py --exp_dir /7_central_fair_notion5_dist --trials 20 --notion 5 --N 7 --dist --exp_params IROS2024_experiments/manhattan_params2.yaml

# # backup2
# python mpc_test.py --exp_dir /10_central_fair_notion0 --trials 40 --notion 0 --N 10 --exp_params IROS2024_experiments/manhattan_params2.yaml
# python mpc_test.py --exp_dir /10_central_fair_notion2 --trials 40 --notion 2 --N 10 --exp_params IROS2024_experiments/manhattan_params2.yaml
# python mpc_test.py --exp_dir /10_central_fair_notion4 --trials 40 --notion 4 --N 10 --exp_params IROS2024_experiments/manhattan_params2.yaml
# python mpc_test.py --exp_dir /10_central_fair_notion5 --trials 40 --notion 5 --N 10 --exp_params IROS2024_experiments/manhattan_params2.yaml


# python mpc_test.py --exp_dir /10_central_fair_notion0_dist --trials 20 --notion 0 --N 10 --dist --exp_params IROS2024_experiments/manhattan_params2.yaml
# python mpc_test.py --exp_dir /10_central_fair_notion2_dist --trials 20 --notion 2 --N 10 --dist --exp_params IROS2024_experiments/manhattan_params2.yaml
# python mpc_test.py --exp_dir /10_central_fair_notion4_dist --trials 20 --notion 4 --N 10 --dist --exp_params IROS2024_experiments/manhattan_params2.yaml
# python mpc_test.py --exp_dir /10_central_fair_notion5_dist --trials 20 --notion 5 --N 10 --dist --exp_params IROS2024_experiments/manhattan_params2.yaml


# # backup 1
# python mpc_test.py --exp_dir /3_central_fair_notion3 --trials 40 --notion 3 --N 3 --exp_params IROS2024_experiments/manhattan_params2.yaml
# python mpc_test.py --exp_dir /5_central_fair_notion3 --trials 40 --notion 3 --N 5 --exp_params IROS2024_experiments/manhattan_params2.yaml
# python mpc_test.py --exp_dir /7_central_fair_notion3 --trials 40 --notion 3 --N 7 --exp_params IROS2024_experiments/manhattan_params2.yaml

# python mpc_test.py --exp_dir /3_central_fair_notion3_dist --trials 20 --notion 3 --N 3 --dist --exp_params IROS2024_experiments/manhattan_params2.yaml
# python mpc_test.py --exp_dir /5_central_fair_notion3_dist --trials 20 --notion 3 --N 5 --dist --exp_params IROS2024_experiments/manhattan_params2.yaml
# python mpc_test.py --exp_dir /7_central_fair_notion3_dist --trials 20 --notion 3 --N 7 --dist --exp_params IROS2024_experiments/manhattan_params2.yaml

# python mpc_test.py --exp_dir /10_central_fair_notion3 --trials 40 --notion 3 --N 10 --exp_params IROS2024_experiments/manhattan_params2.yaml

# python mpc_test.py --exp_dir /10_central_fair_notion3_dist --trials 20 --notion 3 --N 10 --dist --exp_params IROS2024_experiments/manhattan_params2.yaml
