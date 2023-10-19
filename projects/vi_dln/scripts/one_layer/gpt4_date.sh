set -x  # print commands to terminal
dataset=date_understanding
p_class_tpl="classify_forward:3.0"
iters=20
batch_size=10
num_p_samples=15
bwd_temp=0.7
held_out_prompt_ranking=True
use_memory=2
tolerance=2
num_h_samples=5
q_prompt_tpl="q_action_prompt:v3.5"
logp_penalty=2.
posterior_temp=1.
fwd_model_type="gpt-4"

dir=log/one_layer_gpt_4_temp_0.7_new/${dataset}
/bin/rm -rf ${dir}

for seed in 13 42 25; do
    python vi_main.py \
        --balance_batch \
        --one_layer \
        --num_p_samples ${num_p_samples} \
        --bwd_temp ${bwd_temp} \
        --iters ${iters} \
        --q_prompt ${q_prompt_tpl} \
        --p_class ${p_class_tpl} \
        --out_dir ${dir} \
        --batch_size ${batch_size} \
        --seed ${seed} \
        --dataset ${dataset} \
        --use_memory ${use_memory} \
        --tolerance ${tolerance} \
        --held_out_prompt_ranking ${held_out_prompt_ranking} \
        --output_scoring_function accuracy \
        --fwd_model_type ${fwd_model_type}
done
