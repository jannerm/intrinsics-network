base=nips_components/
save=1

for train_set in car_vector motorbike_vector,airplane_vector;
do
    for lr in 0.01; 
    do
        path=${base}vector_shader_${train_set}_${lr}
        echo $lr $path
        sbatch -c 2 --qos=tenenbaum --gres=gpu:titan-x:1 --mem=40G --time=6-12:0 -J shad${lr} shader.lua \
            -lr ${lr} -save_path ${path} -save_model ${save} \
            -train_sets ${train_set} -val_sets car_vector,airplane_vector,motorbike_vector \
            -array_path ../dataset/arrays/shader2.npy
    done
done