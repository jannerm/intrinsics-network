path=components/vector_depth_state

# for data in airplane_vector,bottle_vector,car_vector motorbike_vector,bottle_vector,car_vector motorbike_vector,airplane_vector,car_vector
for data in motorbike_vector airplane_vector car_vector bottle_vector
do 
    for lr in 0.1
    do
        save_path=${path}_${data}_shader_${lr}
        echo ${save_path}
        sbatch -c 6 --qos=tenenbaum --gres=gpu:titan-x:1 --mem=60G -J sh_${data} --time=3-12:0 shader.py \
            --lr ${lr} --save_path ${save_path} --train_sets ${data}
        # for lights_mult in 1
        # do
            # save_path=${path}_decomp_${data}_${lr}lr_${lights_mult}lights
            # echo ${save_path}
            # sbatch -c 6 --qos=tenenbaum --gres=gpu:titan-x:1 --mem=60G --time=3-12:0 decomposer.py \
            #     --lr ${lr} --save_path ${save_path} --lights_mult ${lights_mult} \
            #     --array shader2 --set_size 10000 \
            #     --train_sets ${data}
        # done
    done
done