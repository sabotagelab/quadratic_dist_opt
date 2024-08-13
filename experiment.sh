#!/usr/bin/env bash
#SBATCH -t 3-12:30:00
#SBATCH -J fairfly						  # name of job
#SBATCH -A hlab	  # name of my sponsored account, e.g. class or research group, NOT ONID!
#SBATCH -p athena								  # name of partition or queue
#SBATCH -o fairfly.out				  # name of output file for this submission script
#SBATCH -e fairfly.err				  # name of error file for this submission script
#SBATCH -c 28
#SBATCH --mem=75G

source /nfs/stak/users/frondan/.bashrc
conda activate quad_dist_opt

# EXPERIMENT 1
python mpc_test_fixed_pos_rand_obs.py --exp_dir /exp1_central_no_fair --notion 2 --trials 200 --N 5 &> /exp1_central_no_fair/test.out
python mpc_test_fixed_pos_rand_obs.py --exp_dir /exp1_central_fair0 --notion 0 --trials 200 --N 5 &> /exp1_central_fair0/test.out
python mpc_test_fixed_pos_rand_obs.py --exp_dir /exp1_central_fair3 --notion 3 --trials 200 --N 5 &> /exp1_central_fair3/test.out
python mpc_test_fixed_pos_rand_obs.py --exp_dir /exp1_central_fair4 --notion 4 --trials 200 --N 5 &> /exp1_central_fair4/test.out
python mpc_test_fixed_pos_rand_obs.py --exp_dir /exp1_central_fair5 --notion 5 --trials 200 --N 5 &> /exp1_central_fair5/test.out

python mpc_test_fixed_pos_rand_obs.py --exp_dir /exp1_dist_no_fair --notion 2 --trials 200 --N 5 &> /exp1_dist_no_fair/test.out
python mpc_test_fixed_pos_rand_obs.py --exp_dir /exp1_dist_fair0 --notion 0 --trials 200 --N 5 &> /exp1_dist_fair0/test.out
python mpc_test_fixed_pos_rand_obs.py --exp_dir /exp1_dist_fair3 --notion 3 --trials 200 --N 5 &> /exp1_dist_fair3/test.out
python mpc_test_fixed_pos_rand_obs.py --exp_dir /exp1_dist_fair4 --notion 4 --trials 200 --N 5 &> /exp1_dist_fair4/test.out
python mpc_test_fixed_pos_rand_obs.py --exp_dir /exp1_dist_fair5 --notion 5 --trials 200 --N 5 &> /exp1_dist_fair5/test.out

# EXPERIMENT 2
python mpc_test_fixed_obs_scale_num_drones.py --exp_dir /exp2_dist_no_fair_N5 --notion 2 --trials 20 --N 5 --dist &> /exp2_dist_no_fair_N5/test.out
python mpc_test_fixed_obs_scale_num_drones.py --exp_dir /exp2_dist_fair0_N5 --notion 0 --trials 20 --N 5 --dist &> /exp2_dist_fair0_N5/test.out
python mpc_test_fixed_obs_scale_num_drones.py --exp_dir /exp2_dist_fair3_N5 --notion 3 --trials 20 --N 5 --dist &> /exp2_dist_fair3_N5/test.out
python mpc_test_fixed_obs_scale_num_drones.py --exp_dir /exp2_dist_fair4_N5 --notion 4 --trials 20 --N 5 --dist &> /exp2_dist_fair4_N5/test.out
python mpc_test_fixed_obs_scale_num_drones.py --exp_dir /exp2_dist_fair5_N5 --notion 5 --trials 20 --N 5 --dist &> /exp2_dist_fair5_N5/test.out

python mpc_test_fixed_obs_scale_num_drones.py --exp_dir /exp2_dist_no_fair_N7 --notion 2 --trials 20 --N 7 --dist &> /exp2_dist_no_fair_N7/test.out
python mpc_test_fixed_obs_scale_num_drones.py --exp_dir /exp2_dist_fair0_N7 --notion 0 --trials 20 --N 7 --dist &> /exp2_dist_fair0_N7/test.out
python mpc_test_fixed_obs_scale_num_drones.py --exp_dir /exp2_dist_fair3_N7 --notion 3 --trials 20 --N 7 --dist &> /exp2_dist_fair3_N7/test.out
python mpc_test_fixed_obs_scale_num_drones.py --exp_dir /exp2_dist_fair4_N7 --notion 4 --trials 20 --N 7 --dist &> /exp2_dist_fair4_N7/test.out
python mpc_test_fixed_obs_scale_num_drones.py --exp_dir /exp2_dist_fair5_N7 --notion 5 --trials 20 --N 7 --dist &> /exp2_dist_fair5_N7/test.out

python mpc_test_fixed_obs_scale_num_drones.py --exp_dir /exp2_dist_no_fair_N10 --notion 2 --trials 20 --N 10 --dist &> /exp2_dist_no_fair_N10/test.out
python mpc_test_fixed_obs_scale_num_drones.py --exp_dir /exp2_dist_fair0_N10 --notion 0 --trials 20 --N 10 --dist &> /exp2_dist_fair0_N10/test.out
python mpc_test_fixed_obs_scale_num_drones.py --exp_dir /exp2_dist_fair3_N10 --notion 3 --trials 20 --N 10 --dist &> /exp2_dist_fair3_N10/test.out
python mpc_test_fixed_obs_scale_num_drones.py --exp_dir /exp2_dist_fair4_N10 --notion 4 --trials 20 --N 10 --dist &> /exp2_dist_fair4_N10/test.out
python mpc_test_fixed_obs_scale_num_drones.py --exp_dir /exp2_dist_fair5_N10 --notion 5 --trials 20 --N 10 --dist &> /exp2_dist_fair5_N10/test.out

python mpc_test_fixed_obs_scale_num_drones.py --exp_dir /exp2_dist_no_fair_N12 --notion 2 --trials 20 --N 12 --dist &> /exp2_dist_no_fair_N12/test.out
python mpc_test_fixed_obs_scale_num_drones.py --exp_dir /exp2_dist_fair0_N12 --notion 0 --trials 20 --N 12 --dist &> /exp2_dist_fair0_N12/test.out
python mpc_test_fixed_obs_scale_num_drones.py --exp_dir /exp2_dist_fair3_N12 --notion 3 --trials 20 --N 12 --dist &> /exp2_dist_fair3_N12/test.out
python mpc_test_fixed_obs_scale_num_drones.py --exp_dir /exp2_dist_fair4_N12 --notion 4 --trials 20 --N 12 --dist &> /exp2_dist_fair4_N12/test.out
python mpc_test_fixed_obs_scale_num_drones.py --exp_dir /exp2_dist_fair5_N12 --notion 5 --trials 20 --N 12 --dist &> /exp2_dist_fair5_N12/test.out

python mpc_test_fixed_obs_scale_num_drones.py --exp_dir /exp2_dist_no_fair_N15 --notion 2 --trials 20 --N 15 --dist &> /exp2_dist_no_fair_N15/test.out
python mpc_test_fixed_obs_scale_num_drones.py --exp_dir /exp2_dist_fair0_N15 --notion 0 --trials 20 --N 15 --dist &> /exp2_dist_fair0_N15/test.out
python mpc_test_fixed_obs_scale_num_drones.py --exp_dir /exp2_dist_fair3_N15 --notion 3 --trials 20 --N 15 --dist &> /exp2_dist_fair3_N15/test.out
python mpc_test_fixed_obs_scale_num_drones.py --exp_dir /exp2_dist_fair4_N15 --notion 4 --trials 20 --N 15 --dist &> /exp2_dist_fair4_N15/test.out
python mpc_test_fixed_obs_scale_num_drones.py --exp_dir /exp2_dist_fair5_N15 --notion 5 --trials 20 --N 15 --dist &> /exp2_dist_fair5_N15/test.out
