base=saved_logs_normed/
mkdir ${base};
save=0

original=bottle
new=car

mults="10:0,1,0,5"
lr=0.001;

for num in {1..10};
do
    path=${base}log_${original}_${new}_${mults}_${lr}_${num}
    echo $lr $mults $path
    sbatch  -c 2 --gres=gpu:titan-x:1 --mem=55G --time=6-12:0 -J log_${mults} composer.lua \
            -lr ${lr} -multipliers ${mults} -save_path ${path} -save_model ${save} \
            -train_sets ${new}_normalized -test_path ${new}_normalized -base_class ${original}
done