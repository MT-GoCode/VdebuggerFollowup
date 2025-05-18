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

import re

stage_two_execution_log_filepath = "/local3/minhtr/VIPER/VdebuggerFollowup/results/FULL_LV_BENCH_6AM_5_16/stage_two_execution_log.txt"

result_title_regex = re.compile(r"^-+\s*RESULT\s*-+$")
sample_title_regex = re.compile(r"^=+\s*SAMPLE ID \d+\s*=+$")

all_results = []

with open(stage_two_execution_log_filepath, "r", encoding="utf-8") as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    if result_title_regex.match(lines[i].strip()):
        result_block = []
        i += 1
        while i < len(lines) and not sample_title_regex.match(lines[i].strip()):
            line = lines[i].strip()
            if line:
                result_block.append(line)
            i += 1
        all_results.append(" ".join(result_block))  # join into one string
    else:
        i += 1

def my_collate(batch):
    # Avoid stacking images (different size). Return everything as a list (avoid tensorizing batches)
    to_return = {k: [d[k] for d in batch] for k in batch[0].keys()}
    return to_return


def prepare_dataset(dataset_config):
    from datasets import get_dataset
    dataset = get_dataset(dataset_config)
    return dataset

dataset = prepare_dataset(config.dataset)

dataloader = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=my_collate)

from context import context

context.set_stage(3)

all_answers = []
all_possible_answers = []
for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
    all_answers.append(batch['answer'][0])
    all_possible_answers.append(batch['possible_answers'][0])

print(f"Final Accuracy: {dataset.accuracy(all_results, all_answers)}")
print(f"Proportion of Answers actually from the answer choices: {len([_ for i, _ in enumerate(all_results) if _ in all_possible_answers[i]])/len(all_answers)}")