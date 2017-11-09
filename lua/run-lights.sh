base=saved/
save=1

for lr in 0.1 0.01; 
do
    path=${base}normalized_lights_${lr}
    echo $lr $path
    sbatch -c 2 --qos=tenenbaum --gres=gpu:titan-x:1 --mem=40G --time=6-12:0 -J l${lr} lights.lua -lr ${lr} -save_path ${path} -save_model ${save}
done