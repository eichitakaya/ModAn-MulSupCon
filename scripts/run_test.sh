# model_numを100, 200, ... , 1000として実行
for i in {100..1000..100}; do
    python result_test.py --model_num $i
done