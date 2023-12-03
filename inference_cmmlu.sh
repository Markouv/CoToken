#!/bin/bash

# CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run --nproc_per_node 1 --master_port 48765 \
# inference_llama_pt.py --ckpt_dir /139-4t/private/radoth/downloaded_models/Llama-2-7b-chat --tokenizer_path /139-4t/private/radoth/downloaded_models/Llama-2-7b-chat/tokenizer.model \
# --mode classification \
# --dataset cmmlu --subset_id 0 --func_load_path checkpoints_1125/2023-11-25-epoch_10/checkpoint50000.pth --logits_bias 10.0

GPU_LIST_1=(4)
GPU_LIST_2=(5)
BASE_PORT=58765

for i in {0..66..1}
do

for j in {0..0}
do

if [ $((i+j)) -gt 66 ]
then
break
fi

master_port=$((BASE_PORT+$i+$j))

CUDA_VISIBLE_DEVICES=${GPU_LIST_1[$j]},${GPU_LIST_2[$j]} \
python -m torch.distributed.run --nproc_per_node 1 --master_port $master_port \
inference_llama_pt.py --ckpt_dir /139-4t/private/radoth/downloaded_models/Llama-2-7b-chat --tokenizer_path /139-4t/private/radoth/downloaded_models/Llama-2-7b-chat/tokenizer.model \
--mode classification_with_judge \
--dataset cmmlu --subset_id $(($i+$j)) --func_load_path checkpoints_1125/2023-11-25-epoch_10/checkpoint50000.pth --logits_bias 10.0 &

echo "CUDA_VISIBLE_DEVICES=${GPU_LIST_1[$j]},${GPU_LIST_2[$j]} $(($i+$j)) &"

done
wait
done

echo "Done!"
