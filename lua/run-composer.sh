base=saved_composer_fixed/
mkdir ${base};
save=1

original=bottle
new=car
# albedo, normals, lights, supervised
# mults="30:0,1,0,5-5:.0001,0,0,.00001"
mults="10:0,1,0,50"
# for num in {1..10};
# do
for lr in 0.000001 0.0000001 0.00000001 0.000000001; 
do
    path=${base}${original}_${new}_${mults}_${lr}_noqos_${num}
    echo $lr $mults $path
    sbatch  -c 2 --gres=gpu:titan-x:1 --mem=55G --time=6-12:0 -J ${mults} composer.lua \
            -lr ${lr} -multipliers ${mults} -save_path ${path} -save_model ${save} \
            -train_sets ${new}_normalized -test_path ${new}_sirfs -base_class ${original} -test_size 500 \
            -labeled_sets ${original}_normalized
done
# done

# current run: bottle --> {car, motorbike, boat}
# albedo that works: 
# .1, .01 at .01
# .0001, .00001 at .01