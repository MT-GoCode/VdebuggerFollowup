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

# main_batch.py can set up the output queues to listen to function results, but vision_models, which sets up the forward calls, must be the one to set up input queues.

worker_log_path = None

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


def run_program(dataset_function_parameters, timeout, log_to_stdout, function_body_sample):
    from image_patch import ImagePatch, llm_query, best_image_match, distance, bool_to_yesno, coerce_to_numeric
    from video_segment import VideoSegment

    function_body, sample = function_body_sample

    sample_id = sample['id']
    auxiliary_string = sample.get('auxiliary_string', None)

    
    import time
    import os
    from functools import partial

    # SAMPLE HEADER
    print_section('=', f"SAMPLE ID {sample_id}", "", worker_log_path, log_to_stdout)

    # Build and log code
    function_header = (
        f"def execute_command("
        f"{dataset_function_parameters}, "
        "ImagePatch, VideoSegment, llm_query, bool_to_yesno, distance, best_image_match, coerce_to_numeric):\n"
    )
    code = function_header + function_body
    print_section('-', f"PREPARED CODE FOR {sample_id}", code, worker_log_path, log_to_stdout)

    # Write code to temp file
    modulename = f"{sample_id}_{str(uuid.uuid4())}"
    filename = f"{modulename}.py"
    with open(filename, 'w') as f:
        f.write(code)

    x = importlib.import_module(modulename)
    time.sleep(5)

    dataset_function_arguments = {key: sample[key] for key in sample if key not in ("id", "auxiliary_string")}

    try:
        # for multiprocessing
        queues=(model_processes_input_queues, worker_reusable_output_queue)

        args_dict = {
            **dataset_function_arguments,
            "ImagePatch": partial(ImagePatch, queues = queues),
            "VideoSegment": partial(VideoSegment, queues = queues),
            "llm_query": partial(llm_query, queues=queues),
            "bool_to_yesno": bool_to_yesno,
            "distance": distance,
            "best_image_match": best_image_match,
            "coerce_to_numeric": coerce_to_numeric
        }

    except NameError as e: # singleprocessing
        args_dict = {
            **dataset_function_arguments,
            "ImagePatch": ImagePatch,
            "VideoSegment": VideoSegment,
            "llm_query": llm_query,
            "bool_to_yesno": bool_to_yesno,
            "distance": distance,
            "best_image_match": best_image_match,
            "coerce_to_numeric": coerce_to_numeric
        }

    import threading
    import traceback
    import os
    import sys

    result = None
    exc_info = None  # Will hold exception info from thread if any

    def run_command():
        nonlocal result, exc_info
        try:
            result = x.execute_command(**args_dict)
        except Exception:
            exc_info = sys.exc_info()  # (exc_type, exc_value, traceback)

    try:
        # EXECUTION + ARGUMENT LOGGING
        print_section('-', f"SAMPLE ID: {sample_id} INVOCATION", f"{filename}.execute_command(**args_dict)", worker_log_path, log_to_stdout)
        arg_string = "\n".join(f"{k}: {repr(v)}" for k, v in args_dict.items())
        print_section('-', f"SAMPLE ID: {sample_id} ARGUMENTS", arg_string, worker_log_path, log_to_stdout)

        # Run command in a separate thread with timeout
        thread = threading.Thread(target=run_command)
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            # Timeout occurred
            thread.join(1)  # Give short grace period
            print_section('!', "TIMEOUT ERROR", f"Sample {sample_id} timed out after {timeout} seconds", worker_log_path, log_to_stdout)
            result = f"timeout after {timeout} seconds"

        elif exc_info is not None:
            # Exception occurred in the thread
            etype, evalue, tb = exc_info
            formatted_tb = ''.join(traceback.format_exception(etype, evalue, tb))
            print_section('!', "RUNTIME ERROR", f"Sample {sample_id} failed with error: {evalue}. \n Traceback: \n{formatted_tb}", worker_log_path, log_to_stdout)
            result = "error during execution"

    finally:
        os.remove(filename)

    print_section('-', f"SAMPLE ID: {sample_id} RESULT", result, worker_log_path, log_to_stdout)

    print_section('-', f"SAMPLE ID: {sample_id} AUXILIARY INFO", auxiliary_string, worker_log_path, log_to_stdout)

    return result, code

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

    # this will only executed in main process, as if name == main is only true for main process
    mp.set_start_method('spawn')

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
    import vision_processes

    vision_processes.initialize_models_and_processes()
    
    all_results = []

    global worker_log_path
    worker_log_path = os.path.join(out_dir, "stage_two_execution_log.txt")
    full_results_json_path = os.path.join(out_dir, "stage_execution_results.json")
    result = {}
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=my_collate)
    
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        sample = {key: batch[key][0] for key in batch}
        
        code = stage_generation_results[sample['id']]['code']
        
        code_return, code = run_program(
                    dataset_function_parameters=config.dataset.stage_execution.function_parameters,
                    function_body_sample=(code, sample),
                    timeout=stage_execution_config.timeout_s,
                    log_to_stdout=True)
        
        result[sample['id']] = {
            "code": code,
            "code_return": code_return,
        }

        all_results.append(code_return)
            
    with open(full_results_json_path, "w") as f_json:
        json.dump(result, f_json, indent=2, ensure_ascii=False, default=str) # the default is important, for non-serializable types, to just try to force to string.

    return result

def worker_init(model_processes_input_queues_, n_workers, worker_reusable_output_queues, out_dir):  
    # Use modulo to map process identity to valid worker index - very important! we already have other processes.
    identity = (mp.current_process()._identity[0] - 1) % n_workers

    global model_processes_input_queues
    model_processes_input_queues = model_processes_input_queues_

    global worker_reusable_output_queue
    worker_reusable_output_queue = worker_reusable_output_queues[identity]

    global worker_log_path
    worker_log_path = os.path.join(out_dir, f"stage_execution_log_worker{identity}.txt")

def stage_execution_multiprocessing(stage_execution_config, dataset, stage_generation_results, out_dir):
    all_sample_ids, all_function_bodies = zip(*[(id_, value['code']) for id_, value in stage_generation_results.items()])

    from vision_processes import initialize_models_and_processes, finish_all_consumers

    manager = mp.Manager()
    model_process_process_directory = dict()
    model_processes_input_queues = dict()

    initialize_models_and_processes(manager, model_process_process_directory, model_processes_input_queues) 

    # worker output queue setup
    n_workers = stage_execution_config.multiprocessing.num_workers
    
    worker_reusable_output_queues = [manager.Queue() for _ in range(n_workers)]

    def dataset_stream(dataset_):
        for i in range(len(dataset_)):
            yield dataset_[i]

    # Pass worker index to each worker during initialization
    with mp.Pool(
        processes=n_workers,
        initializer=worker_init,
        initargs=(model_processes_input_queues, n_workers, worker_reusable_output_queues, out_dir)
    ) as pool:

        results = list(
            tqdm(
                pool.imap(
                    partial(
                        run_program,
                        config.dataset.stage_execution.function_parameters,
                        stage_execution_config.timeout_s,
                        False
                    ),
                    zip(all_function_bodies, dataset_stream(dataset))
                ),
                total=len(dataset),
                desc="RUNNING SAMPLES"
            )
        )

    finish_all_consumers(model_processes_input_queues, model_process_process_directory)

    import pdb; pdb.set_trace()

    result = {}
    for sample_id, (code_return, code) in zip(all_sample_ids, results): 
        result[sample_id] = {
            "code": code,
            "code_return": code_return,
        }

    full_results_json_path = os.path.join(out_dir, "stage_execution_results.json")
    with open(full_results_json_path, "w") as f_json:
        json.dump(result, f_json, indent=2, ensure_ascii=False, default=str)

    return result
    
def stage_evaluation(stage_evaluation_config, dataset, stage_execution_results, out_dir):

    eval_json_path = os.path.join(out_dir, "stage_execution_evaluation.json")

    all_answers = []
    all_code_returns = []

    for _, stage_execution_sample_result in stage_execution_results.items():
        all_code_returns.append(stage_execution_sample_result['code_return'])

    all_answers = [sample['answer'] for sample in dataset]

    accuracy = dataset.accuracy(all_code_returns, all_answers)
    print(f'Final accuracy: {accuracy}')

    total_examples = len(all_answers)
    total_execution_errors = 0

    for code_return in all_code_returns:
        normal_code_return = str(code_return).lower()
        if "error during execution" in normal_code_return:
            total_execution_errors += 1

    evaluation = {
        "overall_accuracy": accuracy,
        "total_examples": total_examples,
        "total_error_during_execution": total_execution_errors
    }

    with open(eval_json_path, "w") as f_json:
        json.dump(evaluation, f_json, indent=2, ensure_ascii=False)

    
def deserialize_stage_generation_results(path):
    import json
    with open(path, 'r') as f:
        return json.load(f)
    
# different but duplicate function in case you wanted to change in the future
def deserialize_stage_execution_results(path):
    import json
    with open(path, 'r') as f:
        return json.load(f)

from context import context 

if __name__ == '__main__':
    trial_path = config_check_and_init()
    
    dataset = prepare_dataset(config.dataset)

    stage_generation_results = None

    if config.stage_generation.enabled:
        context.stage = 'generation'
        stage_generation_results = stage_generation(config.stage_generation, dataset, trial_path)
    
    stage_execution_results = None

    if config.stage_execution.enabled:
        if stage_generation_results is None: 
            stage_generation_results = deserialize_stage_generation_results(config.stage_execution.stage_generation_results_path)

        context.stage = 'execution'
        if not config.stage_execution.multiprocessing.use:
            stage_execution_results = stage_execution(config.stage_execution, dataset, stage_generation_results, trial_path)
        else:
            stage_execution_results = stage_execution_multiprocessing(config.stage_execution, dataset, stage_generation_results, trial_path)

    if config.stage_evaluation.enabled:
        if stage_execution_results is None: 
            if config.stage_generation.enabled:
                print("skipping evaluation because it seems generation was done, then execution was skipped. You may have an execution json in config.stage_evaluation.stage_execution_results_path but there isn't much point to this.")
                exit(1)

            stage_execution_results = deserialize_stage_execution_results(config.stage_evaluation.stage_execution_results_path)

        context.stage = 'evaluation'
        stage_evaluation(config.stage_evaluation, dataset, stage_execution_results, trial_path)

    # remember child processes will always re-import everything even if they start elsewhere. so we need to make sure certain things only happen in main thread
    # key idea: we want to create mp queues and shared data structures in main, put only pickleable items in there, and create vanilla encompassing datastructures that contain references to these structures for bookeeping -- but keep that only in main. through reimports, spawned child processes should not accidentally recreate these.