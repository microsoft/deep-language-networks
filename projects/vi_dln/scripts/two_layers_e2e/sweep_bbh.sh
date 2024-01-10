set -x  # print commands to terminal
dataset=${1}
iters=50
batch_size=20
num_p_samples=20
use_memory=5
tolerance=5
num_h_samples=5
p_class_tpl="classify_forward:v3.0"
q_prompt_tpl="q_action_prompt:v3.5"
p_hidden_tpl="suffix_forward_tbs"
q_hidden_tpl='"suffix_forward_tbs_y|suffix_forward_tbs"'

# if dataset == logical_deduction_seven_objects, set this to 512
if [ ${dataset} == "logical_deduction_seven_objects" ]; then
    p1_max_tokens=512
else
    p1_max_tokens=256
fi

# sweep space
bwd_temps=(0.7)
posterior_temps=(1.0)
logp_decays=(False True)
num_examples=(-1)
batch_sizes=(20)
trust_factors=(0 5)
held_out_prompt_ranking=(False True)
one_layer=False
fwd_model_type="text-davinci-003"
bwd_model_type="text-davinci-003"
log_penalties=(0.0 1.0 3.0 5.0)
train_p2s=(True)
h_max=(False)

# remove temp jobs file
rm -rf /tmp/jobs_${dataset}.txt

for logp_penalty in ${log_penalties[@]}; do
for posterior_temp in ${posterior_temps[@]}; do
for bwd_temp in ${bwd_temps[@]}; do
for batch_size in ${batch_sizes[@]}; do
for decay in ${logp_decays[@]}; do
for num_example in ${num_examples[@]}; do
for tf in ${trust_factors[@]}; do
for train_p2 in ${train_p2s[@]}; do
for hmax in ${h_max[@]}; do
for hout in ${held_out_prompt_ranking[@]}; do

dir=log/one_layer${one_layer}_e2e/${dataset}_new_sweep/${fwd_model_type}_${bwd_model_type}_trp2${train_p2}_hmax${hmax}_tf${tf}_heldoutpromptrank${hout}_nex${num_example}_stripoptFalse_decay${decay}_logp${logp_penalty}_bwdt${bwd_temp}_bsz${batch_size}_np${num_p_samples}_nh${num_h_samples}_pt${posterior_temp}
/bin/rm -rf ${dir}

for seed in 13 42 25; do
    echo "python vi_main.py \
        --rewrite_loss_only False \
        --val_freq 2 \
        --do_first_eval \
        --one_layer ${one_layer} \
        --train_p1 True \
        --train_p2 ${train_p2} \
        --balance_batch \
        --p1_max_tokens ${p1_max_tokens} \
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
        --held_out_prompt_ranking ${hout} \
        --trust_factor ${tf} \
        --fwd_model_type ${fwd_model_type} \
        --bwd_model_type ${bwd_model_type} \
        --forward_use_classes True \
        --logp_penalty ${logp_penalty} \
        --posterior_temp ${posterior_temp} \
        --use_h_argmax ${hmax} \
        --decay_logp_penalty ${decay}" >> /tmp/jobs_${dataset}.txt
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
done

# launch
# head -n 1 /tmp/jobs_${dataset}.txt
parallel -j 15 < /tmp/jobs_${dataset}.txt
