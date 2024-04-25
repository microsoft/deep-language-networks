set -x  # print commands to terminal
dataset=navigate
epochs=50
batch_size=10
seed=42
learning_rate=0.03
num_virtual_tokens=16

# dataset
for dataset in navigate logical_deduction_seven_objects; do

dir=log/phi2/${dataset}/soft-prompts/

if [ ! -f ${dir}/done.txt ]; then
    /bin/rm -rf ${dir}

    # seed
    for learning_rate in 0.1 0.01 0.001; do
        accelerate launch phi2_soft_prompts_multitask.py \
            --learning_rate ${learning_rate} \
            --epochs ${epochs} \
            --seed ${seed} \
            --num_virtual_tokens ${num_virtual_tokens} \
            --enable_wandb False \
            --dataset ${dataset}
    done
fi

# dataset
done