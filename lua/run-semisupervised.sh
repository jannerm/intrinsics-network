base=saved_semi_log/
mkdir ${base}
save=0

# render, albedo, normals, lights, shading
mults="10:0,1,5,1,0-100:1,1,5,1,1"
for sup_size in 25 500 12500 15000 17500;
do
    for lr in 0.01; 
    do
        path=${base}car_${sup_size}_${lr}_${mults}
        echo $lr $mults $path
        sbatch -c 2 --gres=gpu:titan-x:1 --mem=55G --time=6-12:0 -J ${sup_size}_${lr} semisupervised.lua -lr ${lr} -sup_size ${sup_size} -multipliers ${mults} -save_path ${path} -save_model ${save}
    done
done

# albedo that works: 
# .1, .01 at .01
# .0001, .00001 at .01