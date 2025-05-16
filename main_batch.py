import datetime
import math
import os
import pathlib
from functools import partial
import warnings
import traceback
import importlib
import time


import pandas as pd
import torch.multiprocessing as mp
from joblib import Memory
from num2words import num2words
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import config
from utils import seed_everything
import datasets
import json

import uuid

# See https://github.com/pytorch/pytorch/issues/11201, https://github.com/pytorch/pytorch/issues/973
# Not for dataloader, but for multiprocessing batches
mp.set_sharing_strategy('file_system')
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
queue_results = None

cache = Memory('cache/' if config.use_cache else None, verbose=0)
runs_dict = {}
seed_everything()

if config.clear_cache:
    cache.clear()

def my_collate(batch):
    # Avoid stacking images (different size). Return everything as a list (avoid tensorizing batches)
    to_return = {k: [d[k] for d in batch] for k in batch[0].keys()}
    return to_return

def print_section(symbol, title, content, log_file=None, print_to_stdout=True):
    header = f"{symbol * 25} {title} {symbol * max(0, 80 - len(title) - 26)}\n"
    block = f"{header}{content}\n\n"
    
    if print_to_stdout:
        print(block)

    if log_file:
        with open(log_file, 'a') as f:
            f.write(block)


def run_program(dataset_function_parameters, function_body, sample, queues_in_, log_file, timeout = 600):
    from image_patch import ImagePatch, llm_query, best_image_match, distance, bool_to_yesno, coerce_to_numeric
    from video_segment import VideoSegment

    sample_id = sample['id']
    import time
    import os
    from functools import partial

    global queue_results

    # SAMPLE HEADER
    print_section('=', f"SAMPLE ID {sample_id}", "", log_file)

    # Build and log code
    function_header = (
        f"def execute_command("
        f"{dataset_function_parameters}, "
        "ImagePatch, VideoSegment, llm_query, bool_to_yesno, distance, best_image_match, coerce_to_numeric):\n"
        "    # Answer is\n"
    )
    code = function_header + function_body
    print_section('-', f"PREPARED CODE FOR {sample_id}", code, log_file)

    # Write code to temp file
    modulename = f"{sample_id}_{str(uuid.uuid4())}"
    filename = f"{modulename}.py"
    with open(filename, 'w') as f:
        f.write(code)

    x = importlib.import_module(modulename)
    time.sleep(5)

    queues = [queues_in_, queue_results]
    image_patch_partial = partial(ImagePatch, queues=queues)
    video_segment_partial = partial(VideoSegment, queues=queues)
    llm_query_partial = partial(llm_query, queues=queues)

    dataset_function_arguments = {key: sample[key] for key in sample if key not in ("id", "answer")}

    args_dict = {
        **dataset_function_arguments,
        "ImagePatch": image_patch_partial,
        "VideoSegment": video_segment_partial,
        "llm_query": llm_query_partial,
        "bool_to_yesno": bool_to_yesno,
        "distance": distance,
        "best_image_match": best_image_match,
        "coerce_to_numeric": coerce_to_numeric
    }

    import threading
    import traceback
    import os

    try:
        # EXECUTION + ARGUMENT LOGGING
        print_section('-', "INVOCATION", f"{filename}.execute_command(**args_dict)", log_file)
        arg_string = "\n".join(f"{k}: {repr(v)}" for k, v in args_dict.items())
        print_section('-', "ARGUMENTS ", arg_string, log_file)

        result = None
        def run_command():
            nonlocal result
            result = x.execute_command(**args_dict)

        # Run command in a separate thread
        thread = threading.Thread(target=run_command)
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            # Timeout occurred
            thread.join(1)  # Short grace period
            print_section('!', "TIMEOUT ERROR", f"Sample {sample_id} timed out after {timeout} seconds", log_file)
            result = f"timeout after {timeout} seconds"

    except Exception as e:
        tb = traceback.format_exc()
        print_section('!', "RUNTIME ERROR", f"Sample {sample_id} failed with error: {e}. \n Traceback: \n {tb}", log_file)
        result = "error during execution"
    finally:
        os.remove(filename)

    print_section('-', "RESULT", result, log_file)

    return result, code

def worker_init(queue_results_):
    global queue_results
    index_queue = mp.current_process()._identity[0] % len(queue_results_)
    queue_results = queue_results_[index_queue]

def config_check_and_init():
    from dotenv import load_dotenv
    load_dotenv()

    import os
    import sys
    from datetime import datetime
    import pytz
    import re

    # Check if artifact folder exists
    if not os.path.isdir(config.artifact_folder):
        print(f"❌ Error: Artifact folder does not exist: {config.artifact_folder}")
        sys.exit(1)

    # Validate trial_name (if provided)
    if config.trial_name is not None:
        # Ensure it's a valid filename (basic check: alphanum + dash/underscore)
        if not re.match(r'^[\w\-\.]+$', config.trial_name):
            print(f"❌ Error: Invalid trial name: {config.trial_name}")
            sys.exit(1)

        trial_path = os.path.join(config.artifact_folder, config.trial_name)
        if os.path.exists(trial_path):
            print(f"❌ Error: Trial folder already exists: {trial_path}")
            sys.exit(1)
    else:
        # No trial name — generate one with PST date-time
        pst = pytz.timezone('US/Pacific')
        now = datetime.now(pst)
        formatted_time = now.strftime('%-m.%-d.%Y.%-I.%M.%S%p')  # e.g. 5.15.2025.11.03AM
        config.trial_name = formatted_time
        trial_path = os.path.join(config.artifact_folder, config.trial_name)

    os.mkdir(trial_path)
    print(f"✅ Trial folder will be: {trial_path}")

    import shutil
    save_to = os.path.join(trial_path, os.path.basename(config._metadata_path))
    shutil.copy2(config._metadata_path, save_to)
    print(f'saved config to {save_to}')

    return trial_path


def prepare_dataset(dataset_config):
    from datasets import get_dataset
    dataset = get_dataset(dataset_config)
    return dataset

def stage_generation(stage_generation_config, dataloader, out_dir):
    import inspect
    import code_gen_models
    from tqdm import tqdm
    import os
    import json

    # Get model
    code_generation_model_config = stage_generation_config.model
    code_generation_model = [
        cls for _, cls in inspect.getmembers(code_gen_models, inspect.isclass)
        if issubclass(cls, code_gen_models.CodeGenModel) and \
            cls != code_gen_models.CodeGenModel and \
            code_generation_model_config.category == cls.name
    ][0](code_generation_model_config=code_generation_model_config)

    # Load base prompt
    with open(stage_generation_config.model.base_prompt_path) as f:
        base_prompt = f.read().strip()

    # Prepare output files
    txt_out_path = os.path.join(out_dir, "stage_one_generation_log.txt")
    json_out_path = os.path.join(out_dir, "stage_generation_results.json")

    # Clear existing file
    with open(txt_out_path, "w") as f_txt:
        f_txt.write("")

    result = {}

    dataloader = DataLoader(dataset, batch_size=stage_generation_config.model.batch_size, num_workers=0, collate_fn=my_collate)
    n_batches = len(dataloader)
    codes_all = []
    filled_prompts_all = []

    for i, batch in tqdm(enumerate(dataloader), total=n_batches):
        batch_codes, batch_filled_prompts = code_generation_model.generate(base_prompt, batch)

        codes_all.extend(batch_codes)
        filled_prompts_all.extend(batch_filled_prompts)

        print_section('|', f"BATCH {i+1} / {n_batches}", "", log_file=txt_out_path, print_to_stdout=False)
        
        for i, id in enumerate(batch['id']):
            code = batch_codes[i]
            prompt = batch_filled_prompts[i]

            print_section('=', f"SAMPLE ID: {id}", "", log_file=txt_out_path, print_to_stdout=False)
            print_section('-', f"SAMPLE ID: {id} PROMPT", prompt, log_file=txt_out_path, print_to_stdout=False)
            print_section('-', f"SAMPLE ID: {id} CODE", code, log_file=txt_out_path, print_to_stdout=False)

            result[id] = {"code": code}

    with open(json_out_path, "w") as f_json:
        json.dump(result, f_json, indent=2, ensure_ascii=False)

    return result


def stage_execution(stage_execution_config, dataset, stage_generation_results, out_dir):
    from vision_processes import queues_in, finish_all_consumers, manager

    # RUN CODE
    
    all_results = []
    all_answers = []

    log_file_path = os.path.join(out_dir, "stage_two_execution_log.txt")
    full_results_json_path = os.path.join(out_dir, "stage_execution_results.json")
    result = {}
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=my_collate)
    
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        sample = {key: batch[key][0] for key in batch}

        if not stage_execution_config.multiprocessing:
            code = stage_generation_results[sample['id']]['code']
            
            code_return, code = run_program(dataset_function_parameters=config.dataset.stage_execution.function_parameters,
                        function_body=code,
                        sample=sample,
                        timeout=stage_execution_config.timeout_s,
                        queues_in_ = queues_in,
                        log_file=log_file_path)
            
            result[sample['id']] = {
                "code": code,
                "code_return": code_return,
            }

            all_results.append(code_return)
            all_answers.append(sample['answer'])
            
    with open(full_results_json_path, "w") as f_json:
        json.dump(result, f_json, indent=2, ensure_ascii=False)

    # EVALUATION

    eval_json_path = os.path.join(out_dir, "stage_execution_evaluation.json")

    accuracy = dataset.accuracy(all_results, all_answers)
    print(f'Final accuracy: {accuracy}')

    total_examples = len(dataloader)
    total_execution_errors = 0

    for code_return in all_results:
        normal_code_return = str(code_return).lower()
        if "error during execution" in normal_code_return:
            total_execution_errors += 1

    evaluation = {
        "overall_accuracy": accuracy,
        "total_examples": total_examples,
        "total_error_during_execution": total_execution_errors,
        "data": result
    }

    with open(eval_json_path, "w") as f_json:
        json.dump(evaluation, f_json, indent=2, ensure_ascii=False)

    finish_all_consumers()

def deserialize_stage_generation_results(path):
    import json
    with open(path, 'r') as f:
        return json.load(f)

from context import context


if __name__ == '__main__':
    trial_path = config_check_and_init()
    
    context.set_stage(1)
    dataset = prepare_dataset(config.dataset)

    if config.stage_generation.enabled:
        stage_generation_results = stage_generation(config.stage_generation, dataset, trial_path)
        if config.stage_execution.enabled:
            context.set_stage(2)
            stage_execution(config.stage_execution, dataset, stage_generation_results, trial_path)
    elif config.stage_execution.enabled:
        stage_generation_results = deserialize_stage_generation_results(config.stage_execution.stage_generation_results_path)
        context.set_stage(2)
        stage_execution(config.stage_execution, dataset, stage_generation_results, trial_path)