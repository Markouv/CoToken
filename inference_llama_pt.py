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
from inference_modes import classification_inference, classification_inference_judge
from funchub.math import *
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM


MAX_GEN_LEN = 512

MMLU_SUBJECTS = [
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

CMMLU_SUBJECTS = ['agronomy', 'anatomy', 'ancient_chinese', 'arts', 'astronomy', 'business_ethics', 'chinese_civil_service_exam', 'chinese_driving_rule', 'chinese_food_culture', 'chinese_foreign_policy', 'chinese_history', 'chinese_literature', 
'chinese_teacher_qualification', 'clinical_knowledge', 'college_actuarial_science', 'college_education', 'college_engineering_hydrology', 'college_law', 'college_mathematics', 'college_medical_statistics', 'college_medicine', 'computer_science',
'computer_security', 'conceptual_physics', 'construction_project_management', 'economics', 'education', 'electrical_engineering', 'elementary_chinese', 'elementary_commonsense', 'elementary_information_and_technology', 'elementary_mathematics', 
'ethnology', 'food_science', 'genetics', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_geography', 'high_school_mathematics', 'high_school_physics', 'high_school_politics', 'human_sexuality',
'international_law', 'journalism', 'jurisprudence', 'legal_and_moral_basis', 'logical', 'machine_learning', 'management', 'marketing', 'marxist_theory', 'modern_chinese', 'nutrition', 'philosophy', 'professional_accounting', 'professional_law', 
'professional_medicine', 'professional_psychology', 'public_relations', 'security_study', 'sociology', 'sports_science', 'traditional_chinese_medicine', 'virology', 'world_history', 'world_religions']

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


def load_cot_models(cot_dict, local_rank, cuda_st_idx = 1, temperature=0, top_p=0.95, max_gen_len=MAX_GEN_LEN, ):
    cot_models = {}
    for cot_name, cot_info in cot_dict.items():
        if cot_info["type"] == "baseline":
            cot_models[cot_name] = cot_info["path"]
        elif cot_info["type"] == "url":
            cot_models[cot_name] = cot_info["path"] + "<url>"
        # elif cot_info["type"] == "chatglm2":
        #     torch.cuda.set_device(cuda_st_idx)
        #     tokenizer_chatglm2 = AutoTokenizer.from_pretrained(cot_info["path"], trust_remote_code=True)
        #     model_chatglm2 = AutoModel.from_pretrained(cot_info["path"], trust_remote_code=True).bfloat16().cuda()
        #     model_chatglm2 = model_chatglm2.eval()
        #     model_cuda_idx = cuda_st_idx
        #     def generate_chatglm2(prompts):
        #         print(model_cuda_idx)
        #         torch.cuda.set_device(model_cuda_idx)
        #         inputs = tokenizer_chatglm2(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_gen_len).to("cuda")
        #         outputs = model_chatglm2.generate(**inputs, max_new_tokens=max_gen_len, do_sample=False if temperature == 0 else True, top_p=top_p, temperature=temperature)
        #         gens = []
        #         for idx in range(len(outputs)):
        #             output = outputs.tolist()[idx]
        #             response = tokenizer_chatglm2.decode(output)
        #             gens.append(response)
        #         torch.cuda.set_device(local_rank)
        #         return gens

        #     cot_models[cot_name] = generate_chatglm2
        
        elif cot_info["type"] == "01-ai":
            torch.cuda.set_device(cuda_st_idx)
            tokenizer_Yi = AutoTokenizer.from_pretrained(cot_info["path"], trust_remote_code=True)
            model_Yi = AutoModelForCausalLM.from_pretrained(cot_info["path"], trust_remote_code=True).bfloat16().cuda()
            model_Yi = model_Yi.eval()
            model_cuda_idx = cuda_st_idx
            def generate_Yi(prompts):
                print(model_cuda_idx)
                torch.cuda.set_device(model_cuda_idx)
                inputs = tokenizer_Yi(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_gen_len).to("cuda")
                outputs = model_Yi.generate(**inputs, max_new_tokens=max_gen_len, do_sample=False if temperature == 0 else True, top_p=top_p, temperature=temperature)
                gens = []
                for idx in range(len(outputs)):
                    output = outputs.tolist()[idx]
                    response = tokenizer_Yi.decode(output)
                    gens.append(response)
                torch.cuda.set_device(local_rank)
                return gens

            cot_models[cot_name] = generate_Yi
        else:
            raise NotImplementedError()
        if cot_info["path"] != "baseline" and cot_info["type"] != "url":
            print(cuda_st_idx)
            cuda_st_idx += 1
    torch.cuda.set_device(local_rank)
    return cot_models


def main(ckpt_dir: str, tokenizer_path: str, temperature: float = 0, top_p: float = 0.95, mode: str = "baseline", dataset = "original", return_top: int = 5, logits_bias: float = 0, func_load_path: str = "None", st_idx=0, ed_idx=10000, suffix="", subset_id: int=0):
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
        "general": """Question: [QUESTION]\n
Answer: """,
        "summary": """Question: [QUESTION]\n
Thoughts: [THOUGHTS]\n
Answer: """,
        "judge": """Question: [QUESTION_TEXT]\n\nAnswer: [ANSWER_TEXT]\n\nBased on the question, please judge the given answer's correctness. If the answer is correct, please write 'T', otherwise, please write 'F'.\n\nJudgement (T/F): """,
        "<en-CoT>": """Question: [QUESTION]\nPlease think step by step and give the answer.\n
Answer: """, 
        "<zh-CoT>": """[Round 1]\n\n问：[QUESTION]\n请一步步思考并给出答案。\n\n答：""",
        "<en-CoT>-1": """Question: [QUESTION]\nPlease think step by step and give the answer.\n
Please write down your answer in the format (A), (B), (C), or (D) at the end of your response.\n
Answer: """,
        "<zh-CoT>-1": """Question: [QUESTION]\nPlease think step by step and give the answer.\n
Please write down your answer in the format (A), (B), (C), or (D) at the end of your response.\n
Answer: """
#         "<zh-CoT>-1": """问: [QUESTION]\n请一步步思考并给出答案。\n
# 请将答案用以下格式 (A), (B), (C), (D) 写在回复的最后。\n
# Answer: """
    }

    test_cases = []
    if dataset == "mmlu":
        ds = datasets.load_from_disk(os.path.join('/139-4t/share/evaluation/datasets/lmeval/cais_mmlu/', MMLU_SUBJECTS[subset_id]))
        ds = ds['test']
        texts, options = ds["question"], ds["choices"]
        test_cases = ["{}\nA. {}\nB. {}\nC. {}\nD. {}".format(text, *option) for text, option in zip(texts, options)]
        answers = ds["answer"]
    elif dataset == "cmmlu":
        ds = datasets.load_from_disk(os.path.join('/139-4t/private/radoth/evaluation/cmmlu/', CMMLU_SUBJECTS[subset_id]))
        ds = ds['test']
        texts, options = ds["Question"], list(zip(ds["A"], ds["B"], ds["C"], ds["D"]))
        test_cases = ["以下是关于{}的单项选择题，请直接给出正确答案的选项。\n\n{}\nA. {}\nB. {}\nC. {}\nD. {}".format(CMMLU_SUBJECTS[subset_id], text, *option) for text, option in zip(texts, options)]
        answers = ds["Answer"]
    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented")

    ## for debugging
    # test_cases = test_cases[:1]

    max_gen_len = MAX_GEN_LEN
    func_dict = json.load(open("./CoT_dict_old.json"))


    funcmodel = load(ckpt_dir, tokenizer_path, local_rank, world_size, func_load_path=func_load_path, func_dict=func_dict)
    funcmodel.set_bias(logits_bias)
    funcmodel.eval()

    cot_dict = {
        "<en-CoT>": {"path": "baseline", "type": "baseline"},
        # "<zh-CoT>": {"path": '/139-4t/share/evaluation/models/hf-chatglm2-6b', "type": "chatglm2"},
        "<zh-CoT>": {"path": '/139-4t/private/radoth/downloaded_models/Yi-6B-chat', "type": "01-ai"},
        "<en-CoT>-1": {"path": "gpt-3.5-turbo-1106", "type": "url"},
        "<zh-CoT>-1": {"path": "gpt-3.5-turbo-1106", "type": "url"},
    }
    if local_rank == 0:
        cot_models = load_cot_models(cot_dict, local_rank, world_size, temperature=temperature, top_p=top_p, max_gen_len=max_gen_len)

    for case_idx, question in tqdm(enumerate(test_cases), total=len(test_cases)):
        if case_idx < st_idx:
            continue
        if case_idx >= ed_idx:
            break

        if mode == "classification":
            log = classification_inference(templates, case_idx, question, funcmodel, temperature, top_p, max_gen_len, return_top, cot_models=cot_models)
            log['answer'] = answers[case_idx]
        elif mode == "classification_with_judge":
            log = classification_inference_judge(templates, case_idx, question, funcmodel, temperature, top_p, max_gen_len, return_top, cot_models=cot_models, question_text=texts[case_idx], option_texts=options[case_idx])
            log['answer'] = answers[case_idx]
        else:
            raise NotImplementedError(f"Mode {mode} not implemented")

        if local_rank == 0:
            try:
                func_model_name = func_load_path.split('/')[-1].split('.')[0]
            except:
                func_model_name = func_load_path

            output_dir = f"outputs/{dataset}-judge"
            os.makedirs(output_dir, exist_ok=True)

            with open(f"{output_dir}/inference-{size}-{func_model_name}-{mode}-{dataset}-{subset_id}-bias_{logits_bias}{suffix}.jsonl", "a", encoding='utf-8') as f:
                f.write(json.dumps(log, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    fire.Fire(main)