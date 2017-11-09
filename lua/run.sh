base=nonorm_saved_components/
mkdir ${base};
save=1

class=car_normalized,motorbike_normalized,chair_normalized,bottle_normalized,airplane_normalized
# class=car1

for lr in 0.01; 
do
    path=${base}${class}_channels_${lr}
    echo $lr $path
    sbatch  -c 2 --qos=tenenbaum --gres=gpu:titan-x:1 --mem=40G --time=6-12:0 -J ch${lr} main.lua \
            -lr ${lr} -save_path ${path} -save_model ${save} -train_sets ${class}
done

#### lights

for lr in 0.01; 
do
    path=${base}${class}_lights_${lr}
    echo $lr $path
    sbatch  -c 2 --qos=tenenbaum --gres=gpu:titan-x:1 --mem=40G --time=6-12:0 -J l${lr} lights.lua \
            -lr ${lr} -save_path ${path} -save_model ${save} -train_sets ${class}
done

## shader

for lr in 0.01; 
do
    path=${base}${class}_shader_${lr}
    echo $lr $path
    sbatch  -c 2 --qos=tenenbaum --gres=gpu:titan-x:1 --mem=40G --time=6-12:0 -J shad${lr} shader.lua \
            -lr ${lr} -save_path ${path} -save_model ${save} -train_sets ${class}
done