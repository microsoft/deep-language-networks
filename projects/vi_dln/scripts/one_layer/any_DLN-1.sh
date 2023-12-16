set -x  # print commands to terminal
p_class_tpl="classify_forward:3.0"
iters=20
batch_size=20
num_p_samples=10
bwd_temp=0.7
held_out_prompt_ranking=True
use_memory=2
tolerance=2
num_h_samples=5
q_prompt_tpl="q_action_prompt:v3.5"
logp_penalty=2.
posterior_temp=1.
fwd_model_type="text-davinci-003"
bwd_model_type="text-davinci-003"
one_layer=True

# dataset
for dataset in navigate subj logical_deduction_seven_objects; do

dir=log/any/${dataset}/DLN-1/

if [ ! -f ${dir}/done.txt ]; then
    /bin/rm -rf ${dir}

    # seed
    for seed in 13 42 25; do
        python vi_main.py \
            --balance_batch \
            --one_layer ${one_layer} \
            --do_first_eval \
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
            --output_scoring_function logprobs \
            --fwd_model_type ${fwd_model_type} \
            --bwd_model_type ${bwd_model_type}
    # seed
    done

    touch ${dir}/done.txt
fi
# dataset
done
