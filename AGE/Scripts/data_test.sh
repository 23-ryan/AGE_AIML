cd ..
declare -a modelType=("base_AGE" "base_GEN" "base_ETH" "AGE_ETH_GEN" "AGE_G_GEN" "ETH_G_GEN")
declare -a batchL=("32" "128" "None")
declare -a lrL=("0.001" "0.0005" "0.0001")

# declare -a modelType=("base_GEN")
# declare -a batchL=("32")
# declare -a lrL=("0.001")

for model in "${modelType[@]}"
do
    for i in "${batchL[@]}"
    do
        for j in "${lrL[@]}"
        do
                python3 test.py --model $model --batch_size $i --lr $j --dropout 0.5

        done
    done
done
