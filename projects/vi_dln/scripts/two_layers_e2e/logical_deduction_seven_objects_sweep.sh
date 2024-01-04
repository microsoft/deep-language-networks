set -x  # print commands to terminal
dataset=logical_deduction_seven_objects
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
rm -rf /tmp/jobs_${dataset}.txt

for logp_penalty in 0. 1. 3. 5.; do
for posterior_temp in 1.0; do
for batch_size in 20; do
for num_p_samples in 20; do
for num_h_samples in 5; do
for strip in False; do
for decay in False; do

dir=log/two_layers_e2e/${dataset}/stripopt${strip}_decay${decay}_logp${logp_penalty}_bsz${batch_size}_np${num_p_samples}_nh${num_h_samples}_pt${posterior_temp}
/bin/rm -rf ${dir}

for seed in 13 42 25; do
    echo "python vi_main.py \
        --val_freq 1 \
        --do_first_eval \
        --num_p1_steps ${num_p1_steps} \
        --train_p1 True \
        --train_p2 True \
        --balance_batch \
        --p1_max_tokens 768 \
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
        --strip_options_for_hidden ${strip} \
        --decay_logp_penalty ${decay}" >> /tmp/jobs_${dataset}.txt
#seed
done
done
done
done
done
done
done

# launch
parallel -j 15 < /tmp/jobs_${dataset}.txt