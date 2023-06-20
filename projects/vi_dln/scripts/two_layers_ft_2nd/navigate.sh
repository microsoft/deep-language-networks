set -x  # print commands to terminal
best_weights_file_path="one-layer-dln-hp-search-result.json"
dataset=navigate
p_class_tpl="classify_forward:3.0"
iters=20
batch_size=20
num_p_samples=20
num_h_samples=5
bwd_temp=0.7
use_h_argmax=False
trust_factor=0.25
deactivate_vi=False
held_out_prompt_ranking=True

q_prompt_tpl="q_action_prompt:v3.5"
logp_penalty=2.
use_memory=2
tolerance=2
p_hidden_tpl="suffix_forward_tbs"
q_hidden_tpl="suffix_backward_h_np_y"

dir=log/two_layers_ft_2nd/${dataset}
/bin/rm -rf ${dir}

for seed in 13 42 25; do
    python vi_main.py \
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
        --use_h_argmax ${use_h_argmax} \
        --use_memory ${use_memory} \
        --tolerance ${tolerance} \
        --held_out_prompt_ranking ${held_out_prompt_ranking} \
        --trust_factor ${trust_factor} \
        --train_p1 True \
        --train_p2 True \
        --init_p2 ${best_weights_file_path} \
        --logp_penalty ${logp_penalty} \
        --forward_use_classes True \
        --do_first_eval
done
