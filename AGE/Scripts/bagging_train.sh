cd ..
declare -a modelType=("Bagging_GEN" "Bagging_AGE" "Bagging_ETH")
declare -a N_EST=("2" "4" "8" "16" "32")

for model in "${modelType[@]}"
do
    for n_est in "${N_EST[@]}"
    do
        index=$(( $n_est-1 ))
        for i in $( eval echo {0..$index} ); do python3 train.py --model $model --epoch 8 --n_estimators 1; mv Saved_Models/Bagging/$model\_1_0.keras Saved_Models/Bagging/$model\_$n_est_$i.keras; done;
    done
done
