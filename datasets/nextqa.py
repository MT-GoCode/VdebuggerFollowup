import json
import os

import pandas as pd
from torch.utils.data import Dataset
import decord
from decord import cpu, gpu
import numpy as np

import numpy as np
from context import context

def load_file(file_name):
    annos = None
    if os.path.splitext(file_name)[-1] == '.csv':
        return pd.read_csv(file_name)
    with open(file_name, 'r') as fp:
        if os.path.splitext(file_name)[1]== '.txt':
            annos = fp.readlines()
            annos = [line.rstrip() for line in annos]
        if os.path.splitext(file_name)[1] == '.json':
            annos = json.load(fp)

    return annos


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


class NExTQADataset():
    def __init__(self, **kwargs):

        assert kwargs["version"] in ['openended', 'multiplechoice']
        directory = 'nextqa' if kwargs["version"] == 'multiplechoice' else 'nextoe'

        self.data_path = kwargs["data_path"]
        self.list_path = kwargs["list_path"]
        self.version = kwargs["version"]

        # sample_list_path = os.path.join(self.data_path, directory, f'{split}.csv')
        self.list_path = os.path.expandvars(self.list_path)
        self.list_path = os.path.expanduser(self.list_path)
        self.sample_list = pd.read_csv(self.list_path, dtype = str)

        if kwargs["shuffle"]:
            self.sample_list = self.sample_list.sample(frac=1, random_state=kwargs["shuffle_seed"]).reset_index(drop=True)
        
        if kwargs["max_samples"] is not None:
            start_sample = kwargs.get('start_sample',0)
            end = start_sample+kwargs["max_samples"]
            print(f'Subset requested. Only selecting from {start_sample} to {end}')
            # self.sample_list = self.sample_list.sample(n=max_samples)
            self.sample_list = self.sample_list[start_sample:end]
        print(self.sample_list)


        self.sample_ids = self.sample_list.index
        self.sample_id_to_index = {sample_id: idx for idx, sample_id in enumerate(self.sample_ids)}

        self.video_to_dir = {}
        
        self.data_path = os.path.expandvars(self.data_path)
        self.data_path = os.path.expanduser(self.data_path)

        for directory in os.listdir(self.data_path):
            for video in os.listdir(os.path.join(self.data_path, directory)):
                self.video_to_dir[video.split('.')[0]] = directory


        self.sample_fps = kwargs['stage_execution']["sample_fps"]
        self.max_num_frames = kwargs['stage_execution']["max_num_frames"]

    def get_sample_path(self, index):
        sample_id = self.sample_ids[index]
        cur_sample = self.sample_list.loc[sample_id]
        video_name = str(cur_sample['video'])
        video_path = os.path.join(self.data_path, 'videos', self.video_to_dir[video_name], video_name + '.mp4')
        return video_path

    def get_video(self, video_path):
        # If fixed width and height are required, VideoReader takes width and height as arguments.
        video_reader = decord.VideoReader(video_path, num_threads=1, ctx=cpu(0))
        decord.bridge.set_bridge('torch')
        vlen = len(video_reader)
        original_fps = video_reader.get_avg_fps()
        num_frames = int(vlen * self.sample_fps / original_fps)
        num_frames = min(self.max_num_frames, num_frames)
        frame_idxs = np.linspace(0, vlen, num_frames, endpoint=False).astype(np.int64)
        video = video_reader.get_batch(frame_idxs).byte()
        video = video.permute(0, 3, 1, 2)
        return video

    def __getitem__(self, idx):

        sample_id = self.sample_ids[idx]
        cur_sample = self.sample_list.loc[sample_id]

        question = str(cur_sample['question'])

        if self.version == 'openended':
            answer = str(cur_sample['answer'])
            if self.tokenize:
                answer = self.tokenize(answer)
            possible_answers = ''
        else:  # multiple choice
            answer_idx = int(cur_sample['answer'])
            possible_answers = [str(cur_sample[f'a{i}']) for i in range(5)]
            answer = possible_answers[answer_idx]
        
        if context.stage == 'generation':
            out_dict = {
                "id": str(sample_id),
                
                "query": question,
                "extra_context": str(possible_answers)
            }
        elif context.stage =='execution':
            video_name = str(cur_sample['video'])
            video_path = os.path.join(self.data_path, self.video_to_dir[video_name], video_name + '.mp4')
            video = self.get_video(video_path)

            out_dict = {
                "id": str(sample_id),

                "video": video,
                "query": question,
                "possible_answers": possible_answers,
                
                "auxilary_string": (f"Answer is {answer} of answer choices {possible_answers}"\
                                    + f"Video path: {video_path}")
            }
        elif context.stage == 'evaluation':
            out_dict = {
                "id": str(sample_id),

                "query": question,
                "possible_answers": possible_answers,
                
                "answer": answer
            }

        return out_dict

    def __len__(self):
        return self.sample_list.shape[0]

    def get_index_from_sample_id(self, sample_id):
        return self.sample_id_to_index[sample_id]

    def accuracy(self, prediction, ground_truth):
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

        # if self.version == 'openended':
        #     for p, g, qt in zip(prediction, ground_truth, query_type):
        #         if isinstance(p, list) or isinstance(p, tuple):
        #             p = p[0]  # p[1] is the info dict
        #         if p is None:
        #             print('None case')
        #             p = 'object'  # To select some word
        #         if qt == 'DC' or qt == 'DB':
        #             s = 1 if remove_stop(p) == remove_stop(g) else 0
        #         else:
        #             s = get_wups(remove_stop(p), remove_stop(g), 0)
        #         score += 100 * s
        if self.version == 'multiplechoice':
            score = sum(1 for p, g in zip(prediction, ground_truth) if p == g)

        return score / len(prediction)
