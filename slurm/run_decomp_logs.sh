path=components_logs/decomp_250

train=motorbike_vector
# for data in airplane_vector,bottle_vector,car_vector motorbike_vector,bottle_vector,car_vector motorbike_vector,airplane_vector,car_vector
for val in motorbike_vector airplane_vector 
do 
    for lr in 0.01 
    do
        for lights_mult in 1
        do
            save_path=${path}_${train}-train_${val}-val_${lr}lr_${lights_mult}lights
            echo ${save_path}
            sbatch -c 6 --qos=tenenbaum --gres=gpu:titan-x:1 --mem=60G --time=3-12:0 -J log250 decomposer.py \
                --train_sets ${train} --val_sets ${val} \
                --lr ${lr} --save_path ${save_path} --lights_mult ${lights_mult} \
                --set_size 10000 --array shader2 \
                --num_val 250 --val_offset 10
        done
    done
done