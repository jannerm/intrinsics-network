base=saved_alternate/
save=1

class=bottle

# for lr in 0.1 0.01; 
# do
#     path=${base}${class}_channels_${lr}
#     echo $lr $path
#     sbatch  -c 2 --gres=gpu:titan-x:1 --mem=40G --time=6-12:0 -J ch${lr} alternate_main.lua \
#             -lr ${lr} -save_path ${path} -save_model ${save} -train_sets ${class}_normalized
# done


new=car

mults="10:1,1,1"
# for lr in 0.1 0.01 0.001 0.0001 0.00001; 
lr=0.0001
for num in {6..6};
do
    path=${base}log_${class}_${new}_${mults}_${lr}_${num}
    echo $lr $mults $path
    sbatch  -c 2 --gres=gpu:titan-x:1 --mem=55G --time=6-12:0 -J alt${mults} alternate_composer.lua \
            -lr ${lr} -multipliers ${mults} -save_path ${path} -save_model ${save} \
            -train_sets ${new}_normalized -test_path ${new}_normalized -base_class ${class}
done

# current run: bottle --> {car, motorbike, boat}
# albedo that works: 
# .1, .01 at .01
# .0001, .00001 at .01