import re
import backoff
import json, requests
from json import JSONDecodeError
from requests import ConnectTimeout, ConnectionError, ReadTimeout
from requests.auth import HTTPBasicAuth
from funchub.math import *

def classification_inference(templates, case_idx, question, funcmodel, temperature, top_p, max_gen_len, return_top=5, cot_models=dict(), choices = ["A", "B", "C", "D"]):
    cur_generation = ""
    cur_generation_with_func = ""
    logs = []
    funcmodel.inference_mode = "func_embedding"
    func_map = list(funcmodel.func_dict.keys())

    results = []
    func_calls = []

    prompt = templates["general"].replace("[QUESTION]", question) + cur_generation
    results = funcmodel.generate([prompt], max_gen_len=max_gen_len, temperature=temperature, top_p=top_p, return_top=return_top, stop_token=[funcmodel.tokenizer.eos_id])
    if return_top > 0:
        results, token_log = results
        logs.append(token_log)
    cur_generation = results[0].replace(templates["general"].replace("[QUESTION]", question), "")

    for op in func_map:
        print("cur_generation: \"", cur_generation, "\"\n\nop: \"", op, "\"\n\n")
        if cur_generation.endswith(op+"("):
            cur_generation_with_func = cur_generation
            print("cur_generation_with_func: \"", cur_generation_with_func, "\"\n\n")
            prompt = templates[op].replace("[QUESTION]", question) + cur_generation_with_func
            prompt = prompt.split(op+"(")[0]
            len_prompt = len(prompt)
            if cot_models[op] == "baseline":
                funcmodel.inference_mode = "baseline"
                print("func prompt: \n", prompt, "\n\n")
                results = funcmodel.generate([prompt], max_gen_len=max_gen_len, temperature=temperature, top_p=top_p, return_top=return_top, stop_token=[funcmodel.tokenizer.eos_id])
                funcmodel.inference_mode = "func_embedding"
                if return_top > 0:
                    results, token_log = results
                    logs.append(token_log)
            else:
                results = cot_models[op]([prompt])

            generated = results[0][len_prompt:]
            cur_generation = cur_generation.split(op+"(")[0] + generated
            print("cur_generation: \"", cur_generation, "\"\n\n")
            
            func_calls.append(op)
            break

    funcmodel.inference_mode = "baseline"
    prompt = templates["summary"].replace("[QUESTION]", question).replace("[THOUGHTS]", cur_generation)
    len_prompt = len(prompt)
    print("final prompt: \n", prompt, "\n\n")
    choice_tokens = [funcmodel.tokenizer.encode(c, bos=False, eos=False) for c in choices]
    choice_tokens = sum(choice_tokens, [])
    disable_token = [x for x in range(funcmodel.model.vocab_size) if x not in choice_tokens]
    results = funcmodel.generate([prompt], max_gen_len=1, temperature=temperature, top_p=top_p, return_top=return_top, disable_token=disable_token)
    funcmodel.inference_mode = "func_embedding"
    if return_top > 0:
        results, token_log = results
        logs.append(token_log)
    generated = results[0][len_prompt:]
    cur_generation = cur_generation + generated
        
    log = {
        "case_idx": case_idx,
        "question": question,
        "func_calls": func_calls,
        "generation": cur_generation.replace("\n", "\\n").strip(),
        "status": "success"
    }

    return log

RETRIES=25
MAX_TOKENS = 512
BETA_URL = "http://43.163.219.59:8001/beta"
@backoff.on_exception(backoff.expo, (TypeError, KeyError, JSONDecodeError, ReadTimeout, ConnectionError, ConnectTimeout), max_tries=RETRIES)
def generate_answer_single_turn(context, model, temperature=0):
    """
    context: str
    """
    # print("question context: ")
    # print(context)
    data = {
        "model": model,
        "messages": [{"role": "user", "content": context}],
        "max_tokens": MAX_TOKENS,
        "temperature": temperature,
    }
    data = json.dumps(data)
    completion = requests.post(url=BETA_URL, data=data, auth=HTTPBasicAuth(username="thumt",password="Thumt@2023"), timeout=300).text
    completion = json.loads(completion)
    # print(completion)
    # print()

    return completion["choices"][0]["message"]["content"], completion["usage"]["prompt_tokens"], completion["usage"]["completion_tokens"]


MODELS_LAYERS = 1
def classification_inference_judge(templates, case_idx, question, funcmodel, temperature, top_p, max_gen_len, return_top=5, cot_models=dict(), choices = ["A", "B", "C", "D"], question_text=None):
    if question_text is None:
        question_text = question
    
    url_prompt_tokens, url_completion_tokens = 0, 0
    cur_generation = ""
    cur_generation_with_func = ""
    logs = []
    funcmodel.inference_mode = "func_embedding"
    func_map = list(funcmodel.func_dict.keys())
    func_map += [f"{func}-{i}" for func in func_map for i in range(1, MODELS_LAYERS+1)]

    results = []
    func_calls = []

    prompt = templates["general"].replace("[QUESTION]", question) + cur_generation
    results = funcmodel.generate([prompt], max_gen_len=max_gen_len, temperature=temperature, top_p=top_p, return_top=return_top, stop_token=[funcmodel.tokenizer.eos_id])
    if return_top > 0:
        results, token_log = results
        logs.append(token_log)
    cur_generation = results[0].replace(templates["general"].replace("[QUESTION]", question), "")

    rev_times = 0
    end_flag = False
    while True:
        for op in func_map:
            print("cur_generation: \"", cur_generation, "\"\n\nop: \"", op, "\"\n\n")
            if cur_generation.endswith(op+"("):
                cur_generation_with_func = cur_generation
                print("cur_generation_with_func: \"", cur_generation_with_func, "\"\n\n")
                prompt = templates[op].replace("[QUESTION]", question) + cur_generation_with_func
                prompt = prompt.split(op+"(")[0]
                len_prompt = len(prompt)
                if cot_models[op] == "baseline":
                    funcmodel.inference_mode = "baseline"
                    print("func prompt: \n", prompt, "\n\n")
                    results = funcmodel.generate([prompt], max_gen_len=max_gen_len, temperature=temperature, top_p=top_p, return_top=return_top, stop_token=[funcmodel.tokenizer.eos_id])
                    funcmodel.inference_mode = "func_embedding"
                    if return_top > 0:
                        results, token_log = results
                        logs.append(token_log)
                elif isinstance(cot_models[op], str) and cot_models[op].endswith("<url>"):
                    result = generate_answer_single_turn(prompt, cot_models[op].split("<url>")[0], temperature=temperature)
                    url_prompt_tokens += result[1]
                    url_completion_tokens += result[2]
                    results = [prompt + result[0]]
                else:
                    results = cot_models[op]([prompt])

                generated = results[0][len_prompt:]
                cur_generation_with_func = cur_generation.split(op+"(")[0] + generated
                print("cur_generation: \"", cur_generation_with_func, "\"\n\n")
                
                func_calls.append(op)
                if op.endswith("-"+str(MODELS_LAYERS)):
                    cur_generation = cur_generation_with_func
                    end_flag = True
                    break
                # break

                # judge
                prompt = templates["judge"].replace("[QUESTION_TEXT]", question_text).replace("[ANSWER_TEXT]", cur_generation_with_func)
                len_prompt = len(prompt)
                funcmodel.inference_mode = "baseline"
                print("func prompt: \n", prompt, "\n\n")
                judge_choices = ["T", "F"]
                choice_tokens = [funcmodel.tokenizer.encode(c, bos=False, eos=False) for c in judge_choices]
                choice_tokens = sum(choice_tokens, [])
                disable_token = [x for x in range(funcmodel.model.vocab_size) if x not in choice_tokens]
                results = funcmodel.generate([prompt], max_gen_len=1, temperature=temperature, top_p=top_p, return_top=return_top, disable_token=disable_token)
                funcmodel.inference_mode = "func_embedding"
                if return_top > 0:
                    results, token_log = results
                    logs.append(token_log)
                generated = results[0][len_prompt:]
                print("\n\njudge generated: ", generated, "\n\n")
                if generated == "F":
                    rev_times += 1
                    cur_generation = cur_generation.split(op+"(")[0] + ((op + "-" + str(rev_times) + "(") if op[-2] != '-' else (op[:-1] + str(rev_times) + "("))
                else:
                    cur_generation = cur_generation_with_func
                    end_flag = True
                break
        if end_flag:
            break

    funcmodel.inference_mode = "baseline"
    prompt = templates["summary"].replace("[QUESTION]", question).replace("[THOUGHTS]", cur_generation)
    len_prompt = len(prompt)
    print("final prompt: \n", prompt, "\n\n")
    choice_tokens = [funcmodel.tokenizer.encode(c, bos=False, eos=False) for c in choices]
    choice_tokens = sum(choice_tokens, [])
    disable_token = [x for x in range(funcmodel.model.vocab_size) if x not in choice_tokens]
    results = funcmodel.generate([prompt], max_gen_len=1, temperature=temperature, top_p=top_p, return_top=return_top, disable_token=disable_token)
    funcmodel.inference_mode = "func_embedding"
    if return_top > 0:
        results, token_log = results
        logs.append(token_log)
    generated = results[0][len_prompt:]
    cur_generation = cur_generation + generated
        
    log = {
        "case_idx": case_idx,
        "question": question,
        "func_calls": func_calls,
        "generation": cur_generation.replace("\n", "\\n").strip(),
        "status": "success",
        "url_prompt_tokens": url_prompt_tokens,
        "url_completion_tokens": url_completion_tokens,
    }

    return log


def func_embedding_inference(templates, case_idx, question, funcmodel, temperature, top_p, max_gen_len, return_top=5):
    cur_generation = ""
    cur_generation_with_func = ""
    start_length = []
    end_length = []
    logs = []
    funcmodel.inference_mode = "func_embedding"
    func_map = list(funcmodel.func_dict.keys())
    try:
        results = []
        func_calls = []
        while True:
            prompt = templates["general"].replace("[QUESTION]", question) + cur_generation
            results = funcmodel.generate([prompt], max_gen_len=max_gen_len, temperature=temperature, top_p=top_p, stop_token=[13], return_top=return_top)
            if return_top > 0:
                results, token_log = results
                logs.append(token_log)
            endflag = True
            current_token = 0
            record_tokens = token_log[-1]
            cur_generation = results[0].replace(templates["general"].replace("[QUESTION]", question), "")
            for op in func_map:
                if cur_generation.endswith(op+"("):
                    endflag = False
                    if start_length and end_length:
                        bias = 0
                        # copy the current generation to cur_generation_with_func
                        cur_generation_with_func = cur_generation
                        for i in range(len(start_length)):
                            cur_generation_with_func = cur_generation_with_func[:start_length[i]+bias] +func_calls[i] + cur_generation_with_func[end_length[i]+bias:]
                            bias += len(func_calls[i]) - (end_length[i] - start_length[i])
                    else:
                        cur_generation_with_func = cur_generation
                    prompt = templates[op].replace("[QUESTION]", question) + cur_generation_with_func
                    len_prompt = len(prompt)
                    funcmodel.inference_mode = "baseline"
                    results = funcmodel.generate([prompt], max_gen_len=max_gen_len, temperature=temperature, top_p=top_p, stop_token=[29897, 3892], return_top=return_top)
                    funcmodel.inference_mode = "func_embedding"
                    if return_top > 0:
                        results, token_log = results
                        logs.append(token_log)
                    generated = results[0][len_prompt:]
                    cur_generation += generated
                    args = cur_generation.split(op)[-1].replace("=", "").replace(">", "").replace("((", "(").replace("))", ")")
                    # remove any $ in the args
                    args = args.replace("$", "")
                    # remove , in the args
                    if ", " in args:
                        args = args.replace(", ", ";").replace(",", "").replace(";", ", ")

                    args = args.replace(" ", "")
                    if "(" not in args or ")" not in args:
                        raise Exception("invalid args")
                    # handle %
                    if '%' in args:
                        temp = args.split("(")[1].split(")")[0].split(",")
                        for arg_i, arg in enumerate(temp):
                            # if have percentage, convert to decimal
                            if "%" in arg:
                                arg = arg.replace("%", "").strip()
                                arg = str(float(arg) / 100)
                            temp[arg_i] = arg
                        args = f"({', '.join(temp)})"
                    try:
                        res = eval(f"{op[1:-1]}_{args}")
                        func_calls.append(f"{op}{args} = {res}")
                        start_length.append(len(cur_generation.split(op)[0]))
                        cur_generation = cur_generation.split(op)[0] + str(res)
                        end_length.append(len(cur_generation))

                        # only generate the next token
                        # disable all the numbers
                        prompt = templates["general"].replace("[QUESTION]", question) + cur_generation
                        results = funcmodel.generate([prompt], max_gen_len=1, temperature=temperature, top_p=top_p, stop_token=[13], return_top=return_top, disable_token = [29900, 29896, 29906, 29941, 29946, 29945, 29953, 29955, 29947, 29929]) # disable all the numbers: 0-9
                        if return_top > 0:
                            results, token_log = results
                            logs.append(token_log)
                        cur_generation = results[0].replace(templates["general"].replace("[QUESTION]", question), "")
                    except:
                        # backtrace 
                        current_token += 1
                        decode_token = lambda x: funcmodel.tokenizer.decode(x) if x < 32000 else func_map[x - 32000]
                        cur_generation = cur_generation.split(op)[0] + decode_token(record_tokens[1][current_token][0])
                    break
            if endflag:
                break

        log = {
            "case_idx": case_idx,
            "question": question,
            "func_calls": func_calls,
            "generation": cur_generation.replace("\n", "\\n").strip(),
            "status": "success"
        }

    except Exception as e:
        log = {
            "case_idx": case_idx,
            "question": question,
            "func_calls": func_calls,
            "generation": cur_generation.replace("\n", "\\n").strip(),
            "status": str(e)
        }
    return log


def vh_embedding_inference(case_idx, question, funcmodel, temperature, top_p, max_func_call):
    funcmodel.inference_mode = "func_embedding"
    inputs = question[0]
    disable_funcs = question[1]
    last_func = []
    for _ in range(max_func_call):
        inputs = funcmodel.generate([inputs], max_gen_len=1, temperature=temperature, top_p=top_p,return_top=0, disable_func=disable_funcs + last_func, no_left_parens=True)[0]

        if inputs.endswith(">"):
            inputs = inputs.replace("]<", "] <")
            inputs += '\n'
            last_func = [] if "[WALK]" in inputs.split("\n")[-2] else re.findall(r"\[.*?\]", inputs.split("\n")[-2])
            print("last func", last_func)
        if "[END]" in inputs.split("Plan:")[-1]:
            break
    

    log = {
    "case_idx": case_idx,
    "question": question[0],
    "func_calls": inputs.replace(question[0], "").strip().split("\n"),
    "generation": inputs.replace("\n", "\\n").strip(),
    # no need to return logs
    # "token_log": logs,
    "status": "success"
    }
    return log


def kamel_embedding_inference(templates, case_idx, question, funcmodel, temperature, top_p, max_gen_len, max_func_call):

    funcmodel.inference_mode = "func_embedding"
    cur_generation = ""
    if "funcgeneral" not in templates:
        templates["funcgeneral"] = templates["general"]
    try:
        results = []
        func_calls = []
        while True:
            if max_func_call == 0:
                break
            prompt = templates["funcgeneral"].replace("[QUESTION]", question) + cur_generation

            results = funcmodel.generate([prompt], max_gen_len=max_gen_len, temperature=temperature, top_p=top_p, stop_token=[13])
            max_func_call -= 1
            
            cur_generation = results[0].replace(templates["funcgeneral"].replace("[QUESTION]", question), "")
            # one function token is enough
            break
        log = {
            "case_idx": case_idx,
            "question": question,
            "func_calls": func_calls,
            "generation": cur_generation.replace("\n", "\\n").strip(),
            "status": "success"
        }
        # f.write(json.dumps(log) + "\n")

    except Exception as e:
        # if local_rank == 0:
        log = {
            "case_idx": case_idx,
            "question": question,
            "func_calls": func_calls,
            "generation": cur_generation.replace("\n", "\\n").strip(),
            "status": str(e)
        }
    return log
