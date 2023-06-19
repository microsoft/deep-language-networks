set -x  # print commands to terminal
dataset=subj
p_class_tpl="classify_forward:3.0"
iters=20
batch_size=10
num_p_samples=20
bwd_temp=0.7
held_out_prompt_ranking=True

for q_prompt_tpl in "q_action_prompt:v3.5" ; do
for logp_penalty in 0.1 0.; do
for posterior_temp in 1. ; do
for use_memory in 2 ; do
for tolerance in 2 ; do
for p_hidden_tpl in "analysis_forward" ; do
for q_hidden_tpl in "analysis_backward" ; do
for trust_factor in 1. 0. ; do
for num_h_samples in 5; do

dir=log/two_layers_e2e/${dataset}
/bin/rm -rf ${dir}

for seed in 13 42 25; do
    python vi_main.py \
        --do_first_eval \
        --balance_batch \
        --num_p_samples ${num_p_samples} \
        --num_h_samples ${num_h_samples} \
        --bwd_temp ${bwd_temp} \
        --iters ${iters} \
        --p_hidden ${p_hidden_tpl} \
        --q_hidden ${q_hidden_tpl} \
        --q_prompt ${q_prompt_tpl} \
        --p_class ${p_class_tpl} \
        --out_dir ${dir} \
        --batch_size ${batch_size} \
        --seed ${seed} \
        --dataset ${dataset} \
        --use_memory ${use_memory} \
        --tolerance ${tolerance} \
        --held_out_prompt_ranking ${held_out_prompt_ranking} \
        --trust_factor ${trust_factor} \
        --train_p1 True \
        --train_p2 True \
        --forward_use_classes True \
        --logp_penalty ${logp_penalty} \
        --posterior_temp ${posterior_temp} \
        --strip_options_for_hidden True \
        --strip_prefix_for_hidden False \
        --use_h_argmax True
done
done
done
done
done
done
done
done
done
done