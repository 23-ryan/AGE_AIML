cd ..
# declare -a modelType=("base_GEN" "base_AGE" "base_ETH" "AGE_ETH_GEN" "AGE_G_GEN" "ETH_G_GEN")
declare -a modelTypeBoost=("Boosting_GEN" "Boosting_ETH")
# declare -a modelType=("base_GEN" "base_ETH")
declare -a batchL=("32" "128" "None")
declare -a lrL=("0.0005" "0.0001" "0.001")
declare -a nestL=("5" "10" "15")
declare -a bboL=("128" "256" "512")

# declare -a modelType=("base_GEN")
# declare -a batchL=("None")
# declare -a lrL=("0.001")

# num=0

# for model in "${modelType[@]}"
# do
#     for i in "${batchL[@]}"
#     do
#         for j in "${lrL[@]}"
#         do
#             python3 train.py --model $model --batch_size $i --lr $j --epochs 8
#         done
#     done
# done

for model in "${modelTypeBoost[@]}"
do
    for i in "${nestL[@]}"
    do
        for j in "${bboL[@]}"
        do
            
            python3 train.py --model $model --n_estimators $i --backbone_output $j 

        done
    done
done

