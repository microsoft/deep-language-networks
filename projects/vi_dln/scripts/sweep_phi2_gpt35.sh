# set -x  # print commands to terminal
dataset=${1}
iters=20
held_out_prompt_ranking=True
use_memory=5
tolerance=2
p_class_tpl="classify_forward:3.0"
q_prompt_tpl="q_action_prompt:v3.5"
p_hidden_tpl="suffix_forward_tbs"
q_hidden_tpl="suffix_forward_tbs_y|suffix_forward_tbs"
fwd_model_type="microsoft/phi-2"
bwd_model_type="gpt-35-turbo-instruct"
one_layer=False

# sweep space
batch_sizes=(20)
trust_factors=(5.)
strip_options=(True)
logp_decays=(False)
bwd_temps=(0.7)
posterior_temps=(1.0)
num_p_samples_sweep=(10 20 50)
num_h_samples_sweep=(05 10 20)
log_penalties=(0.0 2.0 5.0 7.0)


for trust_factor in ${trust_factors[@]}; do
for posterior_temp in ${posterior_temps[@]}; do
for bwd_temp in ${bwd_temps[@]}; do
for batch_size in ${batch_sizes[@]}; do
for strip in ${strip_options[@]}; do
for decay in ${logp_decays[@]}; do
for num_p_samples in ${num_p_samples_sweep[@]}; do
for num_h_samples in ${num_h_samples_sweep[@]}; do
for logp_penalty in ${log_penalties[@]}; do

dir=log/one_layer${one_layer}_e2e/${dataset}/fmt${model_type}_bmt${bwd_model_type}_stripopt${strip}_decay${decay}_logp${logp_penalty}_bwdt${bwd_temp}_bsz${batch_size}_np${num_p_samples}_nh${num_h_samples}_pt${posterior_temp}_tf${trust_factor}

for seed in 13 42 25; do
    python vi_main.py \
        --val_freq 2 \
        --do_first_eval \
        --one_layer ${one_layer} \
        --train_p1 True \
        --train_p2 True \
        --balance_batch \
        --p1_max_tokens 512 \
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
        --fwd_model_type ${fwd_model_type} \
        --bwd_model_type ${bwd_model_type} \
        --forward_use_classes True \
        --logp_penalty ${logp_penalty} \
        --posterior_temp ${posterior_temp} \
        --strip_options_for_hidden ${strip} \
        --decay_logp_penalty ${decay}
#seed
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

# launch
# parallel -j 15 < /tmp/jobs_${dataset}.txt
# head -n 1 /tmp/jobs_${dataset}.txt



# hyperbaton navigate date_understanding logical_deduction_seven_objects mpqa trec subj disaster airline

# nohup bash scripts/sweep.sh hyperbaton &
# nohup bash scripts/sweep.sh navigate &
# nohup bash scripts/sweep.sh date_understanding &
# nohup bash scripts/sweep.sh logical_deduction_seven_objects &
# nohup bash scripts/sweep.sh mpqa &
# nohup bash scripts/sweep.sh trec &
# nohup bash scripts/sweep.sh subj &
# nohup bash scripts/sweep.sh disaster &
# nohup bash scripts/sweep.sh airline &