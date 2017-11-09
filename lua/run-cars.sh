## channels
base=saved/airplane
save=1
train_set=airplane_normals_emit_0.05
val_set=car_normals_emit_0.05

for lr in 0.01; 
do
    path=${base}_channels_${lr}
    echo $lr $path
    sbatch -c 2 --gres=gpu:titan-x:1 --mem=40G --time=6-12:0 -J airch${lr} main.lua -lr ${lr} -save_path ${path} -save_model ${save} \
        -train_sets ${train_set} -test_sets ${train_set},${val_set}
done

## lights

for lr in 0.01; 
do
    path=${base}_lights_${lr}
    echo $lr $path
    sbatch -c 2 --qos=tenenbaum --gres=gpu:titan-x:1 --mem=40G --time=6-12:0 -J airl${lr} lights.lua -lr ${lr} -save_path ${path} -save_model ${save} \
        -train_sets ${train_set}
done

## shader

for lr in 0.01; 
do
    path=${base}_shader_${lr}
    echo $lr $path
    sbatch -c 2 --qos=tenenbaum --gres=gpu:titan-x:1 --mem=40G --time=6-12:0 -J airsh${lr} shader.lua -lr ${lr} -save_path ${path} -save_model ${save} \
        -train_sets ${train_set} -val_sets ${train_set},${val_set}
done

## composer

# albedo, normals, lights, supervised
mults="25:0,1,0,.1-15:.0001,0,0,.00001"
for lr in 0.01 0.001 0.0001; 
do
    path=${base}log_${mults}_${lr}
    echo $lr $mults $path
    sbatch -c 2 --qos=tenenbaum --gres=gpu:titan-x:1 --mem=55G --time=6-12:0 -J ${mults}_${sup_mult} composer.lua -lr ${lr} -multipliers ${mults} -save_path ${path} -save_model ${save}
done
