base=saved_semishader_fixed2/
mkdir ${base};
save=0

class=car
# new=car
# albedo, normals, lights, supervised

# mults="10:1,5,1,0-10:.5,1,.5,1-10:.1,.5,.1,1-10:.1,.1,.1,5"
# mults="10:1,5,1,0"

# for mults in 10:0,0,0,1 10:0,0,0,1-10:.5,1,.5,1-10:.1,.5,.1,1-10:.1,.1,.1,5;
for mults in 10:0,0,0,1-100:0,1,0,1;
do
    # 25 50 1000 5000 10000 20000
    for sup_size in 25 50 100 250 500 750 1000 5000;
    do
        for lr in 0.001; 
        do
            path=${base}${class}_${sup_size}_${lr}_${mults}
            echo $lr $mults $path
            sbatch  -c 2 --gres=gpu:titan-x:1 --mem=55G --time=6-12:0 -J s${sup_size}_${mults} semishader.lua \
                    -lr ${lr} -multipliers ${mults} -save_path ${path} -save_model ${save} \
                    -train_sets ${class}_normalized -labeled_sets ${class}_normalized -test_path ${class}_normalized \
                    -test_size 500 -sup_size ${sup_size}
        done
    done
done

# current run: bottle --> {car, motorbike, boat}
# albedo that works: 
# .1, .01 at .01
# .0001, .00001 at .01