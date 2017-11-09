base=saved_composer_real_processed/
save=1

original=car_normalized,motorbike_normalized,chair_normalized,bottle
new=mit_berkeley
# albedo, normals, lights, supervised
# mults="30:0,1,0,5-5:.0001,0,0,.00001"
mults="10:0,1,0,5"
for lr in 0.1 0.01 0.001 0.0001 0.00001; 
do
    path=${base}${original}_${new}_${mults}_${lr}
    echo $lr $mults $path
    sbatch  -c 2 --gres=gpu:titan-x:1 --mem=55G --time=6-12:0 -J mit${mults} composer.lua \
            -lr ${lr} -multipliers ${mults} -save_path ${path} -save_model ${save} \
            -train_sets ${new} -test_path ${new} -base_class ${original} \
            -net_path saved_components_real -test_size 200 -num_val_save 200 \
            -labeled_sets ${original}_normalized
done

# current run: bottle --> {car, motorbike, boat}
# albedo that works: 
# .1, .01 at .01
# .0001, .00001 at .01