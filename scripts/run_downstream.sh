# visualize representation
#python visualize_representation.py


# 全パターンの実験を10回ずつ実行

# scratch
#python downstream_evaluation.py --dataset thyroid --model_num 1 --epochs 10 --batchsize 32 --model_type scratch --num_experiments 10
#python downstream_evaluation.py --dataset breast --model_num 1 --epochs 10 --batchsize 32 --model_type scratch --num_experiments 10
#python downstream_evaluation.py --dataset acl --model_num 1 --epochs 10 --batchsize 32 --model_type scratch --num_experiments 10

# imagenet
#python downstream_evaluation.py --dataset thyroid --model_num 1 --epochs 10 --batchsize 32 --model_type imagenet --num_experiments 10
#python downstream_evaluation.py --dataset breast --model_num 1 --epochs 10 --batchsize 32 --model_type imagenet --num_experiments 10
#python downstream_evaluation.py --dataset acl --model_num 1 --epochs 10 --batchsize 32 --model_type imagenet --num_experiments 10

# radimagenet
python downstream_evaluation.py --dataset thyroid --model_num 1 --epochs 10 --batchsize 32 --model_type radimagenet --num_experiments 10
python downstream_evaluation.py --dataset breast --model_num 1 --epochs 10 --batchsize 32 --model_type radimagenet --num_experiments 10
python downstream_evaluation.py --dataset acl --model_num 1 --epochs 10 --batchsize 32 --model_type radimagenet --num_experiments 10

# simCLR
#python downstream_evaluation.py --dataset thyroid --model_num 1 --epochs 10 --batchsize 32 --model_type simCLR --num_experiments 10
#python downstream_evaluation.py --dataset breast --model_num 1 --epochs 10 --batchsize 32 --model_type simCLR --num_experiments 10
#python downstream_evaluation.py --dataset acl --model_num 1 --epochs 10 --batchsize 32 --model_type simCLR --num_experiments 10

# SSL
# --model_numを100, 200, ... , 1000として実行
#for i in {100..1000..100}; do
#    python downstream_evaluation.py --dataset thyroid --model_num $i --epochs 10 --batchsize 32 --model_type SSL --num_experiments 10
#    python downstream_evaluation.py --dataset breast --model_num $i --epochs 10 --batchsize 32 --model_type SSL --num_experiments 10
#    python downstream_evaluation.py --dataset acl --model_num $i --epochs 10 --batchsize 32 --model_type SSL --num_experiments 10
#done