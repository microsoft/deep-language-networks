set -x  # print commands to terminal
dataset=navigate
iters=20
p_class_tpl="classify_forward:3.0"
q_prompt_tpl="q_action_prompt:v3.5"
batch_size=20
num_p_samples=20
bwd_temp=0.7
held_out_prompt_ranking=True
use_memory=5
tolerance=2
num_h_samples=5
num_p1_steps=1
posterior_temp=1.0
trust_factor=5.
p_hidden_tpl="suffix_forward_tbs"
q_hidden_tpl='"suffix_forward_tbs_y|suffix_forward_tbs"'

# remove temp jobs file
rm -rf /tmp/jobs.txt

for logp_penalty in 0. 1. 3. 5.; do
for posterior_temp in 0.15 0.25 1.0; do
for batch_size in 20; do
for num_p_samples in 20; do
for num_h_samples in 5; do

dir=log/two_layers_e2e_32ex/${dataset}/logp${logp_penalty}_nodecay_bsz${batch_size}_np${num_p_samples}_nh${num_h_samples}_pt${posterior_temp}
/bin/rm -rf ${dir}

for seed in 13 42 25; do
    echo "python vi_main.py \
        --val_freq 2 \
        --num_p1_steps ${num_p1_steps} \
        --train_p1 True \
        --train_p2 True \
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
        --forward_use_classes True \
        --logp_penalty ${logp_penalty} \
        --posterior_temp ${posterior_temp} \
        --strip_options_for_hidden True \
        --strip_prefix_for_hidden False \
        --decay_logp_penalty False" >> /tmp/jobs.txt
#seed
done
done
done
done
done
done
done

# launch
parallel -j 15 < /tmp/jobs.txt