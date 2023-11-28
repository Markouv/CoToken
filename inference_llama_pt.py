# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json
import re
import random
import numpy as np
from pathlib import Path
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from tqdm import tqdm
from llama import ModelArgs, Transformer, Tokenizer, FunctionLM
import datasets
from inference_modes import func_embedding_inference, kamel_embedding_inference, vh_embedding_inference
from funchub.math import *


SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]
SID = int(sys.argv[1])

def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)
    return local_rank, world_size


def load(ckpt_dir: str, tokenizer_path: str, local_rank: int, world_size: int, func_load_path: str, func_dict: dict) -> FunctionLM:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert (
        world_size == len(checkpoints)
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(max_seq_len=2048, max_batch_size=1, **params)
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args).cuda().half()
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)
    funcmodel = FunctionLM(model, tokenizer, func_dict = func_dict, load_path=func_load_path)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return funcmodel


def main(ckpt_dir: str, tokenizer_path: str, temperature: float = 0, top_p: float = 0.95, mode: str = "baseline", dataset = "original", return_top: int = 5, logits_bias: float = 0, func_load_path: str = "None", st_idx=0, ed_idx=10000, suffix=""):
    # set random seed
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(1)
    np.random.seed(1)
    size = ckpt_dir.split("/")[-1]
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, 'w')

    templates = {
        "general": """Question: [QUESTION]
Answer: """,
        "func": """Question: [QUESTION]
Answer: """,
    }

    test_cases = []
    ds = datasets.load_dataset("cais/mmlu", SUBJECTS[SID], split="test")
    texts, options = ds["question"], ds["choices"]
    test_cases = ["{}\nA. {}\nB. {}\nC. {}\nD. {}".format(text, *option) for text, option in zip(texts, options)]

    max_gen_len = 512
    func_dict = json.load(open("data/CoToken/CoT_dict.json"))


    funcmodel = load(ckpt_dir, tokenizer_path, local_rank, world_size, func_load_path=func_load_path, func_dict=func_dict)
    funcmodel.set_bias(logits_bias)
    funcmodel.eval()

    for case_idx, question in tqdm(enumerate(test_cases), total=len(test_cases)):
        if case_idx < st_idx:
            continue
        if case_idx >= ed_idx:
            break
        log = func_embedding_inference(templates, case_idx, question, funcmodel, temperature, top_p, max_gen_len, return_top)

        if local_rank == 0:
            try:
                func_model_name = func_load_path.split('/')[-1].split('.')[0]
            except:
                func_model_name = func_load_path

            output_dir = f"outputs/{dataset}"
            os.makedirs(output_dir, exist_ok=True)

            with open(f"{output_dir}/inference-{size}-{func_model_name}-{mode}-{dataset}-bias_{logits_bias}{suffix}.jsonl", "a") as f:
                f.write(json.dumps(log) + "\n")


if __name__ == "__main__":
    fire.Fire(main)