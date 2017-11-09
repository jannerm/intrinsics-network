path=logs/composer_generic


# for style_mult in 0 1e-6 1e-9 1e-10 1e-11 1e-12 1e-13 1e-14 1e-15
# do
for lr in 0.01
do
    for lab_mult in 0.1
    do
        for correspondence_mult in 0.1
        do
            for relight_mult in 0.1 0.01
            do
                for relight_sigma in 1 1.5 2 2.5
                do
                    save_path=${path}_${lr}lr_${correspondence_mult}corr-mult_${lab_mult}lab-mult_${relight_mult}relight-mult_${relight_sigma}relight-sigma
                    echo ${save_path}
                    sbatch  -c 6 --gres=gpu:titan-x:1 --mem=60G --time=3-12:0 -J ${lr}_${lab_mult} composer.py \
                            --lab_mult ${lab_mult} --correspondence_mult ${correspondence_mult} \
                            --relight_sigma ${relight_sigma} --relight_mult ${relight_mult} \
                            --save_path ${save_path} --lr ${lr}
                done
            done
        done
    done
done
# done