#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node 4 --master_port 1250 \
inference_llama_pt.py --ckpt_dir $LLAMA_CKPTS/30B --tokenizer_path $LLAMA_CKPTS/tokenizer.model --mode func_embedding \
--dataset cais/mmlu --func_load_path checkpoints_1125/2023-11-25-epoch_10/checkpoint50000.pth --logits_bias 0.0

