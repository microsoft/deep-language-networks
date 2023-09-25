set -x  # print commands to terminal
dataset=navigate
p_class_tpl="classify_forward:3.0"
iters=10
batch_size=5
num_p_samples=10
bwd_temp=0.7
held_out_prompt_ranking=True
use_memory=2
tolerance=2
num_h_samples=5
q_prompt_tpl="q_action_prompt:v3.5"
logp_penalty=2.
posterior_temp=1.
trust_factor=5.
p_hidden_tpl="suffix_forward_tbs"
q_hidden_tpl="suffix_forward_tbs_y|suffix_forward_tbs"
# model_type="/llama/7Bf"
model_type="/data/llama2/70Bf"
fwd_max_tokens=512
bwd_max_tokens=1024
p1_max_tokens=512
p2_max_tokens=512

dir=log/{model_type}/two_layers_e2e/${dataset}
# /bin/rm -rf ${dir}


for seed in 42; do
    python vi_main.py \
        --do_first_eval \
        --balance_batch \
        --model_type ${model_type} \
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
        --fwd_max_tokens ${fwd_max_tokens} \
        --bwd_max_tokens ${bwd_max_tokens} \
        --p1_max_tokens ${p1_max_tokens} \
        --p2_max_tokens ${p2_max_tokens} \
        --strip_options_for_hidden True
done

# export TOKENIZER_PATH="/data/models/llama2/70Bf"