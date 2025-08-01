#!/bin/sh

# 全パターンの実験を10回ずつ実行

# scratch
#python linear_probing.py --dataset thyroid --model_num 1 --epochs 10 --batchsize 32 --model_type scratch --num_experiments 10
#python linear_probing.py --dataset breast --model_num 1 --epochs 10 --batchsize 32 --model_type scratch --num_experiments 10
#python linear_probing.py --dataset acl --model_num 1 --epochs 10 --batchsize 32 --model_type scratch --num_experiments 10

# imagenet
#python linear_probing.py --dataset thyroid --model_num 1 --epochs 10 --batchsize 32 --model_type imagenet --num_experiments 10
#python linear_probing.py --dataset breast --model_num 1 --epochs 10 --batchsize 32 --model_type imagenet --num_experiments 10
#python linear_probing.py --dataset acl --model_num 1 --epochs 10 --batchsize 32 --model_type imagenet --num_experiments 10

# radimagenet
#python linear_probing.py --dataset thyroid --model_num 1 --epochs 10 --batchsize 32 --model_type radimagenet --num_experiments 10
#python linear_probing.py --dataset breast --model_num 1 --epochs 10 --batchsize 32 --model_type radimagenet --num_experiments 10
#python linear_probing.py --dataset acl --model_num 1 --epochs 10 --batchsize 32 --model_type radimagenet --num_experiments 10

# simclr
#python linear_probing.py --dataset thyroid --model_num 1 --epochs 10 --batchsize 32 --model_type simCLR --num_experiments 10
#python linear_probing.py --dataset breast --model_num 1 --epochs 10 --batchsize 32 --model_type simCLR --num_experiments 10
#python linear_probing.py --dataset acl --model_num 1 --epochs 10 --batchsize 32 --model_type simCLR --num_experiments 10

# SSL
python linear_probing.py --dataset thyroid --model_num 200 --epochs 10 --batchsize 32 --model_type SSL --num_experiments 10
python linear_probing.py --dataset breast --model_num 200 --epochs 10 --batchsize 32 --model_type SSL --num_experiments 10
python linear_probing.py --dataset acl --model_num 200 --epochs 10 --batchsize 32 --model_type SSL --num_experiments 10