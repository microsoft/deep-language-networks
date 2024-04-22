set -x  # print commands to terminal
p_class_tpl="classify_forward:3.0"
iters=20
batch_size=10
num_p_samples=10
bwd_temp=0.7
held_out_prompt_ranking=True
use_memory=5
tolerance=2
num_h_samples=10
q_prompt_tpl="q_action_prompt:v3.6"
logp_penalty=2.
posterior_temp=1.
trust_factor=5.
p_hidden_tpl="suffix_forward_tbs"
q_hidden_tpl="suffix_forward_tbs_y|suffix_forward_tbs"
fwd_model_type="microsoft/phi-2"
bwd_model_type="microsoft/phi-2"

# If no arguments are provided, use a hardcoded list of datasets
if [ $# -eq 0 ]; then
    set -- hyperbaton navigate date_understanding logical_deduction_seven_objects mpqa trec subj disaster airline
fi

# dataset
for dataset in "$@"; do

dir=log/phi2/${dataset}/DLN-2/

if [ ! -f ${dir}/done.txt ]; then
    /bin/rm -rf ${dir}

    # seed
    for seed in 13 42 25; do
        python vi_main.py \
            --balance_batch \
            --do_first_eval \
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
            --logp_penalty ${logp_penalty} \
            --posterior_temp ${posterior_temp} \
            --output_scoring_function logprobs \
            --fwd_model_type ${fwd_model_type} \
            --bwd_model_type ${bwd_model_type}
    done
fi

# dataset
done