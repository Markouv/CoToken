#!/bin/bash

# CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run --nproc_per_node 1 --master_port 48765 \
# inference_llama_pt.py --ckpt_dir /139-4t/private/radoth/downloaded_models/Llama-2-7b-chat --tokenizer_path /139-4t/private/radoth/downloaded_models/Llama-2-7b-chat/tokenizer.model \
# --mode classification \
# --dataset cmmlu --subset_id 0 --func_load_path checkpoints_1125/2023-11-25-epoch_10/checkpoint50000.pth --logits_bias 10.0

GPU_LIST_1=(0 2)
GPU_LIST_2=(1 3)
BASE_PORT=48765

for i in {0..56..2}
do

for j in {0..1}
do

if [ $((i+j)) -gt 56 ]
then
break
fi

master_port=$((BASE_PORT+$i+$j))

CUDA_VISIBLE_DEVICES=${GPU_LIST_1[$j]},${GPU_LIST_2[$j]} \
python -m torch.distributed.run --nproc_per_node 1 --master_port $master_port \
inference_llama_pt.py --ckpt_dir /139-4t/private/radoth/downloaded_models/Llama-2-7b-chat --tokenizer_path /139-4t/private/radoth/downloaded_models/Llama-2-7b-chat/tokenizer.model \
--mode classification_with_judge \
--dataset mmlu --subset_id $(($i+$j)) --func_load_path checkpoints_1125/2023-11-25-epoch_10/checkpoint50000.pth --logits_bias 10.0 &

echo "CUDA_VISIBLE_DEVICES=${GPU_LIST_1[$j]},${GPU_LIST_2[$j]} $(($i+$j)) &"

done
wait
done

echo "Done!"
