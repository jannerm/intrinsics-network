path=components_generalize/lights_array_

for data in primitive_cube,primitive_sphere,primitive_cylinder,primitive_cone,primitive_torus 
do
    for lr in 0.001
    do
        save_path=${path}_shader_${lr}
        # echo ${save_path}
        # sbatch -c 6 --qos=tenenbaum --gres=gpu:titan-x:1 --mem=60G --time=3-12:0 shader.py --lr ${lr} --save_path ${save_path}
        for lights_mult in 1
        do
            save_path=${path}_decomp_${data}_${lr}lr_${lights_mult}lights
            echo ${save_path}
            sbatch -c 6 --qos=tenenbaum --gres=gpu:titan-x:1 --mem=60G --time=3-12:0 -J ${lights_mult}_${data} decomposer.py \
                --lr ${lr} --save_path ${save_path} --lights_mult ${lights_mult} \
                --train_sets ${data} --val_sets ${data},primitive_suzanne --set_size 10000 \
                --array shader3
        done
    done
done

####

# path=logs_generalize/shape_train_shapes


# # for style_mult in 0 1e-6 1e-9 1e-10 1e-11 1e-12 1e-13 1e-14 1e-15
# # do
# for lr in 0.001
# do
#     for lab_mult in 10
#     do
#         for correspondence_mult in 0 #0.1
#         do
#             for relight_mult in 0 #0.1 0.01
#             do
#                 for relight_sigma in 0 #1 1.5 2 2.5
#                 do
#                     save_path=${path}_${lr}lr_${correspondence_mult}corr-mult_${lab_mult}lab-mult_${relight_mult}relight-mult_${relight_sigma}relight-sigma
#                     echo ${save_path}
#                     sbatch  -c 6 --qos=tenenbaum --gres=gpu:titan-x:1 --mem=60G --time=3-12:0 -J ${lr}_${lab_mult} composer.py \
#                             --lab_mult ${lab_mult} --correspondence_mult ${correspondence_mult} \
#                             --relight_sigma ${relight_sigma} --relight_mult ${relight_mult} \
#                             --save_path ${save_path} --lr ${lr} \
#                             --decomposer components_generalize/lights_decomp_primitive_cube,primitive_sphere,primitive_cylinder,primitive_cone,primitive_torus_0.001lr_0.5lights/state.t7 \
#                             --unlabeled primitive_suzanne,primitive_bunny,primitive_teapot --labeled primitive_cube,primitive_sphere,primitive_cylinder,primitive_cone,primitive_torus \
#                             --val_sets primitive_cube,primitive_sphere,primitive_cylinder,primitive_cone,primitive_torus \
#                             --unlabeled_array shader3 --labeled_array shader3 --num_val 20 --transfer normals,shader
#                 done
#             done
#         done
#     done
# done
# # done







