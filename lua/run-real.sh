base=saved_composer_real_images/
mkdir ${base};
save=1

original=airplane

# albedo, normals, lights, supervised
# mults="30:0,1,0,5-5:.0001,0,0,.00001"
mults="10:0,1,0,50"
for lr in 0.1 0.01 0.001 0.0001 0.00001; 
do
    path=${base}${original}_${mults}_${lr}
    echo $lr $mults $path
    sbatch  -c 2 --gres=gpu:titan-x:1 --mem=55G --time=6-12:0 -J ${mults}_${sup_mult} real_composer.lua \
            -lr ${lr} -multipliers ${mults} -save_path ${path} -save_model ${save} \
            -base_class ${original}
done
