#!/usr/bin/env bash

echo "3 Agents"
python quadratic_distributed.py --N 3
python quadratic_distributed.py --N 3 --alpha 3
python quadratic_distributed.py --N 3 --beta 3
python quadratic_distributed.py --N 3 --alpha 3 --beta 3
python quadratic_distributed.py --N 3 --alpha 3 --beta 3 --eps_bounds 2


echo "5 Agents"
python quadratic_distributed.py --N 5
python quadratic_distributed.py --N 5 --alpha 3
python quadratic_distributed.py --N 5 --beta 3
python quadratic_distributed.py --N 5 --alpha 3 --beta 3
python quadratic_distributed.py --N 5 --alpha 3 --beta 3 --eps_bounds 2

echo "10 Agents"
python quadratic_distributed.py --N 10
python quadratic_distributed.py --N 10 --alpha 3
python quadratic_distributed.py --N 10 --beta 3
python quadratic_distributed.py --N 10 --alpha 3 --beta 3
python quadratic_distributed.py --N 10 --alpha 3 --beta 3 --eps_bounds 2