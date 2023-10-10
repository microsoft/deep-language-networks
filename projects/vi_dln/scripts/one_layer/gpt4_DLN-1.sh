set -x  # print commands to terminal
p_class_tpl="classify_forward:3.0"
q_prompt_tpl="q_action_prompt:v3.5"
iters=20
batch_size=20
num_p_samples=20
bwd_temp=0.7
held_out_prompt_ranking=True
use_memory=2
tolerance=2
num_h_samples=5
logp_penalty=2.
posterior_temp=1.
model_type="gpt-4"

# dataset
for dataset in subj; do

dir=log/gpt4/${dataset}/dln-1-20bsz/

if [ ! -f ${dir}/done.txt ]; then
    /bin/rm -rf ${dir}

    # seed
    for seed in 13 42 25; do
        python vi_main.py \
            --do_first_eval \
            --balance_batch \
            --one_layer \
            --train_p1 False \
            --train_p2 True \
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
            --model_type ${model_type}
    # seed
    done

    touch ${dir}/done.txt
fi
# dataset
done