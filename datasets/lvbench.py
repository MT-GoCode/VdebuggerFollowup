import json
import os
import shutil

import pandas as pd
from torch.utils.data import Dataset
os.environ["DECORD_NUM_THREADS"] = "1"
os.environ["FFMPEG_THREADS"] = "1"
import decord
from decord import cpu
import numpy as np

import numpy as np

from VdebuggerFollowup.context import context

CACHE_DIR = "./video_tensor_iteration_cache"  # Directory to store cached video tensors
import torch

iteration_cache = [
    # 'LLSJrEgOOtw',
    # 'aJI8XTa_DII',
    # 'JlrzSvCsIjE',
    # '2sriHX3PbXw'
]


from tqdm import tqdm

# upgrade: batching
# @memory.cache
from functools import lru_cache
@lru_cache(maxsize=2)
def tensorize_video(video_path, total_frames):

    # determining frame count

    # iteration cache - loading
    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    disk_cache_path = os.path.join(CACHE_DIR, f"{video_filename}_{total_frames}.pt")
    if video_filename in iteration_cache:
        if os.path.exists(disk_cache_path):
            print(f"[iteration-cache] Loading from: {disk_cache_path}")
            return torch.load(disk_cache_path)

    # end

    print(f"{video_path} not found in cache; tensorizing")
    video_reader = decord.VideoReader(video_path, num_threads=20, ctx=cpu(0))
    decord.bridge.set_bridge('torch')
    vlen = len(video_reader)

    frame_idxs = np.linspace(0, vlen, total_frames, endpoint=False).astype(np.int64)
    
    video_chunks = []
    frame_fetch_batch_size = 2000

    H, W, C = video_reader[0].shape

    # pre-creating video tensor significantly improves performance apparently... no O(n^2) torch.catting
    video = torch.empty((len(frame_idxs), C, H, W), dtype=torch.uint8)

    for i in tqdm(range(0, len(frame_idxs), frame_fetch_batch_size), desc="Decoding frames in batches"):
        batch = frame_idxs[i:i+frame_fetch_batch_size]
        video_batch = video_reader.get_batch(batch).byte().permute(0, 3, 1, 2)
        video_chunks.append(video_batch)
        video[i:i+len(batch)] = video_batch

    print(f"tensorization finished with {video.shape} and {video.dtype}.")

    # iteration cache - saving
    if video_filename in iteration_cache:
        print(f"[iteration-cache] Saving to: {disk_cache_path}")
        os.makedirs(CACHE_DIR, exist_ok=True)
        torch.save(video, disk_cache_path)
    # end

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
    def __init__(self, **kwargs):

        self.video_path = kwargs['video_path']
        self.list_path = kwargs['list_path']
        self.sample_fps = kwargs['sample_fps']
        self.max_num_frames = kwargs['max_num_frames']

        self.list_path = os.path.expandvars(self.list_path)
        self.list_path = os.path.expanduser(self.list_path)
        self.sample_list = pd.read_csv(self.list_path, dtype = str)

        if kwargs['shuffle']:
            self.sample_list = self.sample_list.sample(frac=1, random_state=kwargs['shuffle_seed']).reset_index(drop=True)
        
        if kwargs['max_samples'] is not None:
            start_sample = kwargs.get('start_sample',0)
            end = kwargs.get('start_sample',0)+kwargs['max_samples']
            print(f'Subset requested. Only selecting from {start_sample} to {end}')
            self.sample_list = self.sample_list[start_sample:end]
        print(self.sample_list)

        self.sample_ids = self.sample_list.index
        self.video_path = os.path.expandvars(self.video_path)
        self.video_path = os.path.expanduser(self.video_path)

    def get_sample_path(self, index):
        sample_id = self.sample_ids[index]
        cur_sample = self.sample_list.loc[sample_id]
        video_name = str(cur_sample['video'])
        video_path = os.path.join(self.video_path, video_name + '.mp4')
        return video_path

    def __getitem__(self, idx):

        sample_id = self.sample_ids[idx]
        cur_sample = self.sample_list.loc[sample_id]

        question = cur_sample['question']
        answer = cur_sample['answer']
        duration_s = int(round(float(cur_sample['duration'])))
        possible_answers = [str(cur_sample[f'a{i}']) for i in range(4)]
        fps = int(round(float(cur_sample['fps'])))
        resulting_frame_cnt = min(self.max_num_frames, int(duration_s * fps))

        if context.get_stage() == 1:
            out_dict = {
                "id": str(sample_id),
                
                "query": question,
                "possible_answers": str(possible_answers),
                "duration_s": str(duration_s),
                "resulting_frame_cnt": str(resulting_frame_cnt)
            }
        elif context.get_stage() == 2:
            video_name = str(cur_sample['video'])
            video_path = os.path.join(self.video_path, video_name + '.mp4')
            video = tensorize_video(video_path, total_frames = resulting_frame_cnt)
            print(f"video {video_name} completely finished tensorization. if no progress bar shown, this was retreived from LRU RAM cache.")

            out_dict = {
                "id": str(sample_id),

                "video": video,
                "query": question,
                "possible_answers": possible_answers,
                
                "answer": answer
            }

        return out_dict

    def __len__(self):
        return self.sample_list.shape[0]

    def accuracy(self, prediction, ground_truth):
        assert len(prediction) == len(ground_truth)
        score = sum(1 for p, g in zip(prediction, ground_truth) if p == g)
        return score / len(prediction)
    
    