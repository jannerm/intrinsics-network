base=nips_components/
save=1

for lr in 0.1 0.01; 
do
    path=${base}shadows_vector_shader_${lr}
    echo $lr $path
    sbatch  -c 2 --qos=tenenbaum --gres=gpu:titan-x:1 --mem=40G --time=6-12:0 -J shad${lr} shader.lua \
            -lr ${lr} -save_path ${path} -save_model ${save} -setSize 10000 -array_path ../dataset/arrays/spot1.npy \
            -train_sets shadows_vector -val_sets shadows_vector -param_dim 6
done