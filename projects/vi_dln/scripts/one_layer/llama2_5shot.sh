set -x  # print commands to terminal
datasets_list="subj navigate date_understanding logical_deduction_seven_objects hyperbaton mpqa trec disaster airline"
p_class_tpl="classify_forward:3.0"
num_p_samples=10
held_out_prompt_ranking=True
fwd_max_tokens=512
bwd_max_tokens=1024
p2_max_tokens=512

n_shot=5
connections_config=scripts/one_layer/connections.yaml
model_type="meta-llama/Llama-2-70b-chat-hf"

for dataset in $datasets_list; do

    dir=log/${model_type}/one_layer/5_shot/${dataset}

    for seed in 13 42 25; do
        python vi_main.py \
            --connections_config ${connections_config} \
            --one_layer \
            --do_zero_shot \
            --do_first_eval \
            --balance_batch \
            --seed ${seed} \
            --dataset ${dataset} \
            --n_shots ${n_shot} \
            --model_type ${model_type} \
            --num_p_samples ${num_p_samples} \
            --p_class ${p_class_tpl} \
            --out_dir ${dir} \
            --held_out_prompt_ranking ${held_out_prompt_ranking} \
            --train_p1 False \
            --train_p2 False \
            --forward_use_classes True \
            --fwd_max_tokens ${fwd_max_tokens} \
            --bwd_max_tokens ${bwd_max_tokens} \
            --p2_max_tokens ${p2_max_tokens} \
            --strip_options_for_hidden True
    done
done