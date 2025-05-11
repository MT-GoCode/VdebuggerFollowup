import json
import os
import shutil

import pandas as pd
from torch.utils.data import Dataset
os.environ["DECORD_NUM_THREADS"] = "1"
os.environ["FFMPEG_THREADS"] = "1"
import decord
from decord import cpu, gpu
import numpy as np
import spacy

from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import numpy as np

import nltk
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('punkt_tab')

from pywsd.utils import lemmatize_sentence
from collections import Counter
import joblib
from joblib import Memory

CACHE_DIR = "./video_cache"  # Directory to store cached video tensors
memory = Memory(CACHE_DIR, verbose=0)  # Initialize joblib Memory for caching

import torch

# def get_best_gpu(min_memory_required_mb=2000):
#     # Get the number of GPUs available
#     num_gpus = torch.cuda.device_count()

#     # If there are no GPUs available, fallback to CPU
#     if num_gpus == 0:
#         print("No GPUs found. Falling back to CPU.")
#         return torch.device('cpu')

#     # Initialize variables to track the best GPU (least used with enough memory)
#     best_gpu = None
#     max_free_memory = 0

#     for i in range(num_gpus):
#         # Get the total memory and free memory for each GPU
#         total_memory = torch.cuda.get_device_properties(i).total_memory
#         free_memory = torch.cuda.memory_reserved(i) - torch.cuda.memory_allocated(i)

#         # Convert memory to MB
#         free_memory_mb = free_memory / 1024**2

#         # Check if the GPU has enough free memory and is better than the previous best GPU
#         if free_memory_mb >= min_memory_required_mb and free_memory_mb > max_free_memory:
#             max_free_memory = free_memory_mb
#             best_gpu = i

#     # If we found a suitable GPU, return it, otherwise fallback to CPU
#     if best_gpu is not None:
#         print(f"Using GPU {best_gpu} with {max_free_memory} MB of free memory.")
#         return best_gpu
#         # return torch.device(f'cuda:{best_gpu}')
#     else:
#         print("No GPUs with enough memory. Falling back to CPU.")
#         return -1
#         # return torch.device('cpu')


from tqdm import tqdm

# upgrade: batching
# @memory.cache
from functools import lru_cache
@lru_cache(maxsize=2)
def tensorize_video(video_path, fps=30):
    print(f"tensorizing {video_path} ")
    # device = get_best_gpu(min_memory_required_mb=200)
    # print(f"using GPU {device}")
    print("using cpu btw")
    video_reader = decord.VideoReader(video_path, num_threads=20, ctx=cpu(0))
    decord.bridge.set_bridge('torch')

    vlen = len(video_reader)
    original_fps = video_reader.get_avg_fps()
    num_frames = int(vlen * fps / original_fps)
    frame_idxs = np.linspace(0, vlen, num_frames, endpoint=False).astype(np.int64)
    
    video_chunks = []
    frame_fetch_batch_size = 2000

    H, W, C = video_reader[0].shape

    # pre-creating video tensor significantly improves performance apparently... no O(n^2) torch.catting
    video = torch.empty((len(frame_idxs), C, H, W), dtype=torch.uint8)

    print(f"preparations complete. original frames: {vlen}, original fps: {original_fps}, requested fps: {fps}, so final frame count is {len(frame_idxs)} to be processed in batches of {frame_fetch_batch_size}")
    for i in tqdm(range(0, len(frame_idxs), frame_fetch_batch_size), desc="Decoding frames in batches"):
        batch = frame_idxs[i:i+frame_fetch_batch_size]
        video_batch = video_reader.get_batch(batch).byte().permute(0, 3, 1, 2)
        video_chunks.append(video_batch)
        video[i:i+len(batch)] = video_batch

    print(f"tensorization finished with {video.shape} and {video.dtype}. if caching turned on, that may take a while. ")

    return video

def save_file(obj, filename):
    """
    save obj to filename
    :param obj:
    :param filename:
    :return:
    """
    filepath = os.path.dirname(filename)
    if filepath != '' and not os.path.exists(filepath):
        os.makedirs(filepath)
    else:
        with open(filename, 'w') as fp:
            json.dump(obj, fp, indent=4)


class LVBenchDataset(Dataset):
    def __init__(self, split, data_path, list_path, tokenize=None, max_samples=None, version='openended', fps=30,
                 start_sample=0, clear_tensor_cache = False, tensor_cache = None, **kwargs):

        assert version in ['openended', 'multiplechoice']

        self.split = split
        self.data_path = data_path
        self.list_path = list_path
        self.tokenize = tokenize
        self.version = version
        self.fps = fps
        self.input_type = 'video'

        # THE TENSOR CACHE WILL NOT BE USED
        # self.tensor_cache = tensor_cache
        # if (self.tensor_cache):
        #     self.tensor_cache = os.path.expandvars(self.tensor_cache)
        #     self.tensor_cache = os.path.expanduser(self.tensor_cache)
        #     os.makedirs(self.tensor_cache, exist_ok=True)

        # if (clear_tensor_cache and self.tensor_cache):
        #     if os.path.exists(self.tensor_cache):
        #         shutil.rmtree(self.tensor_cache)

            # don't delete the directory itself
            # os.makedirs(self.tensor_cache, exist_ok=True)
        
        self.list_path = os.path.expandvars(self.list_path)
        self.list_path = os.path.expanduser(self.list_path)
        self.sample_list = pd.read_csv(self.list_path, dtype = str)
        
        if max_samples is not None:
            end = start_sample+max_samples
            print(f'Subset requested. Only selecting from {start_sample} to {end}')
            # self.sample_list = self.sample_list.sample(n=max_samples)
            self.sample_list = self.sample_list[start_sample:end]
            print(self.sample_list)


        self.sample_ids = self.sample_list.index

        self.data_path = os.path.expandvars(self.data_path)
        self.data_path = os.path.expanduser(self.data_path)

    def get_sample_path(self, index):
        sample_id = self.sample_ids[index]
        cur_sample = self.sample_list.loc[sample_id]
        video_name = str(cur_sample['video'])
        video_path = os.path.join(self.data_path, video_name + '.mp4')
        return video_path

    def __getitem__(self, idx):
        print("getting new item. let's check RAM usage -- should be appropriately used based on the number of questions per video & batch size.")
        import subprocess
        output = subprocess.check_output(["free", "-h"], text=True)
        print(output)

        sample_id = self.sample_ids[idx]
        cur_sample = self.sample_list.loc[sample_id]

        question = str(cur_sample['question'])
        if self.tokenize:
            question = self.tokenize(question)

        video_name = str(cur_sample['video'])

        # critical difference from nextqa; videos are stored as siblings - just  merge data_path and video name
        video_path = os.path.join(self.data_path, video_name + '.mp4')

        # if self.tensor_cache:
        #     cache_identifier = f"{video_name}_{self.fps}.pt"

        #     cache_path = os.path.join(self.tensor_cache, cache_identifier)
        #     if not os.path.exists(cache_path):
        #         video = tensorize_video(video_path, fps = self.fps)
        #         torch.save(video, cache_path)
        #         print(f"video {video_name} completely finished processing")
        #     else: 
        #         video = torch.load(cache_path)
        #         print(f"video {video_name} tensor fetched from disk")

        video = tensorize_video(video_path, fps = self.fps)
        print(f"video {video_name} completely finished processing")
        
        if self.version == 'openended':
            answer = str(cur_sample['answer'])
            if self.tokenize:
                answer = self.tokenize(answer)
            possible_answers = ''
        else:  # multiple choice
            answer = str(cur_sample['answer'])
            possible_answers = [str(cur_sample[f'a{i}']) for i in range(4)] # only four answers!

        query_type = str(cur_sample['type'])

        out_dict = {"sample_id": sample_id, "answer": answer, "image": video, "query": question, 'pil_img': -1,
                    "query_type": query_type, 'index': idx, 'possible_answers': possible_answers,
                    'extra_context': possible_answers}

        return out_dict

    def __len__(self):
        return self.sample_list.shape[0]

    def accuracy(self, prediction, ground_truth, possible_answers, query_type):
        """
        Args:
            prediction (list): List of predicted answers.
            ground_truth (list): List of ground truth answers.
            possible_answers (list): List of possible answers.
            query_type (list): List of query types
        Returns:
            score (float): Score of the prediction.
        """

        assert len(prediction) == len(ground_truth)
        score = 0

        if self.version == 'openended':
            for p, g, qt in zip(prediction, ground_truth, query_type):
                if isinstance(p, list) or isinstance(p, tuple):
                    p = p[0]  # p[1] is the info dict
                if p is None:
                    print('None case')
                    p = 'object'  # To select some word
                if qt == 'DC' or qt == 'DB':
                    s = 1 if remove_stop(p) == remove_stop(g) else 0
                else:
                    s = get_wups(remove_stop(p), remove_stop(g), 0)
                score += 100 * s
        else:
            nlp = spacy.load('en_core_web_lg')
            for p, g, a in zip(prediction, ground_truth, possible_answers):
                if isinstance(p, list) or isinstance(p, tuple):
                    if len(p) == 2:
                        p = p[0]  # p[1] is the info dict
                    else:  # Multiple predictions
                        all_answers = []
                        for pp in p:
                            if pp not in a:
                                pred_tokens = nlp(pp)
                                a.sort(key=lambda x: pred_tokens.similarity(nlp(x)), reverse=True)
                                pp = a[0]
                            all_answers.append(pp)
                        # Majority vote
                        c = Counter(all_answers).most_common(1)[0]
                        if c[1] == 1:
                            # If no majority, select the middle one
                            p = all_answers[1]
                        else:
                            p = c[0]
                if p not in a:
                    if p is None:
                        print('None case')  # Should not happen
                    else:
                        pred_tokens = nlp(p)
                        a.sort(key=lambda x: pred_tokens.similarity(nlp(x)), reverse=True)
                    p = a[0]
                if p == g:
                    score += 1
        return score / len(prediction)


# Below is code from https://github.com/doc-doc/NExT-OE/blob/main/eval_oe.py

stopwords = "i, me, my, myself, we, our, ours, ourselves, you, you're, you've, you'll, you'd, your, yours, yourself, " \
            "yourselves, he, him, his, himself, she, she's, her, hers, herself, it, it's, its, itself, they, them, " \
            "their, theirs, themselves, what, which, who, whom, this, that, that'll, these, those, am, is, are, was, " \
            "were, be, been, being, have, has, had, having, do, does, did, doing, a, an, the, and, but, if, or, " \
            "because, as, until, while, to, from, of, at, for, with, about, into, through, during, again, further, " \
            "then, here, there, when, where, why, how, all, any, each, most, other, some, such, only, own, so, than, " \
            "too, very, s, t, can, will, just, don, don't, should, should've, now, d, ll, m, o, re, ve, y, ain, " \
            "aren, aren't, couldn, couldn't, didn, didn't, doesn, doesn't, hadn, hadn't, hasn, hasn't, haven, " \
            "haven't, isn, isn't, ma, mightn, mightn't, mustn, mustn't, needn, needn't, shan, shan't, shouldn, " \
            "shouldn't, wasn, wasn't, weren, weren't, won, won't, wouldn, wouldn't"


def remove_stop(sentence):

    words = lemmatize_sentence(sentence)
    words = [w for w in words if not w in stopwords]
    return ' '.join(words)


def wup(word1, word2, alpha):
    """
    calculate the wup similarity
    :param word1:
    :param word2:
    :param alpha:
    :return:
    """
    # print(word1, word2)
    if word1 == word2:
        return 1.0

    w1 = wordnet.synsets(word1)
    w1_len = len(w1)
    if w1_len == 0: return 0.0
    w2 = wordnet.synsets(word2)
    w2_len = len(w2)
    if w2_len == 0: return 0.0

    #match the first
    word_sim = w1[0].wup_similarity(w2[0])
    if word_sim is None:
        word_sim = 0.0

    if word_sim < alpha:
        word_sim = 0.1*word_sim
    return word_sim


def wups(words1, words2, alpha):
    """
    :param pred:
    :param truth:
    :param alpha:
    :return:
    """
    sim = 1.0
    flag = False
    for w1 in words1:
        max_sim = 0
        for w2 in words2:
            word_sim = wup(w1, w2, alpha)
            if word_sim > max_sim:
                max_sim = word_sim
        if max_sim == 0: continue
        sim *= max_sim
        flag = True
    if not flag:
        sim = 0.0
    return sim


def get_wups(pred, truth, alpha):
    """
    calculate the wups score
    :param pred:
    :param truth:
    :return:
    """
    pred = word_tokenize(pred)
    truth = word_tokenize(truth)
    item1 = wups(pred, truth, alpha)
    item2 = wups(truth, pred, alpha)
    value = min(item1, item2)
    return value