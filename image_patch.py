from __future__ import annotations

import numpy as np
import re
import torch
from dateutil import parser as dateparser
from PIL import Image
from rich.console import Console
from torchvision import transforms
from torchvision.ops import box_iou
from typing import Union, List
from word2number import w2n

from utils import show_single_image, load_json
from vision_processes import forward, config

console = Console(highlight=False)

image_patch_api_config = config.stage_execution.image_patch
models_config = config.stage_execution.models

def llm_query(query, long_answer=None, queues=None):
    """Answers a text question using GPT. The input question is always a formatted string with a variable in it.

    Parameters
    ----------
    query: str
        the text question to ask. Must not contain any reference to 'the image' or 'the photo', etc.
    """
    if long_answer:
        return forward(
            process_name='gpt_long_qa',
            question=query,
            queues=queues
        )
    else:
        return forward(
            process_name='gpt_short_qa',
            question=query,
            queues=queues
        )

class ImagePatch:
    """A Python class containing a crop of an image centered around a particular object, as well as relevant
    information.
    Attributes
    ----------
    cropped_image : array_like
        An array-like of the cropped image taken from the original image.
    left : int
        An int describing the position of the left border of the crop's bounding box in the original image.
    lower : int
        An int describing the position of the bottom border of the crop's bounding box in the original image.
    right : int
        An int describing the position of the right border of the crop's bounding box in the original image.
    upper : int
        An int describing the position of the top border of the crop's bounding box in the original image.

    Methods
    -------
    find(object_name: str)->List[ImagePatch]
        Returns a list of new ImagePatch objects containing crops of the image centered around any objects found in the
        image matching the object_name.
    exists(object_name: str)->bool
        Returns True if the object specified by object_name is found in the image, and False otherwise.
    verify_property(property: str)->bool
        Returns True if the property is met, and False otherwise.
    best_text_match(option_list: List[str], prefix: str)->str
        Returns the string that best matches the image.
    simple_query(question: str=None)->str
        Returns the answer to a basic question asked about the image. If no question is provided, returns the answer
        to "What is this?".
    compute_depth()->float
        Returns the median depth of the image crop.
    crop(left: int, lower: int, right: int, upper: int)->ImagePatch
        Returns a new ImagePatch object containing a crop of the image at the given coordinates.
    """

    def __init__(self, image: Union[Image.Image, torch.Tensor, np.ndarray], left: int = None, lower: int = None,
                 right: int = None, upper: int = None, parent_left=0, parent_lower=0,
                 parent_img_patch=None, queues = None):
        """Initializes an ImagePatch object by cropping the image at the given coordinates and stores the coordinates as
        attributes. If no coordinates are provided, the image is left unmodified, and the coordinates are set to the
        dimensions of the image.

        Parameters
        -------
        image : array_like
            An array-like of the original image.
        left : int
            An int describing the position of the left border of the crop's bounding box in the original image.
        lower : int
            An int describing the position of the bottom border of the crop's bounding box in the original image.
        right : int
            An int describing the position of the right border of the crop's bounding box in the original image.
        upper : int
            An int describing the position of the top border of the crop's bounding box in the original image.

        """

        if isinstance(image, Image.Image):
            image = transforms.ToTensor()(image)
        elif isinstance(image, np.ndarray):
            image = torch.tensor(image).permute(1, 2, 0)
        elif isinstance(image, torch.Tensor) and image.dtype == torch.uint8:
            image = image / 255

        if left is None and right is None and upper is None and lower is None:
            self.cropped_image = image
            self.left = 0
            self.lower = 0
            self.right = image.shape[2]  # width
            self.upper = image.shape[1]  # height
        else:
            self.cropped_image = image[:, image.shape[1]-upper:image.shape[1]-lower, left:right]
            self.left = left + parent_left
            self.upper = upper + parent_lower
            self.right = right + parent_left
            self.lower = lower + parent_lower

        self.height = self.cropped_image.shape[1]
        self.width = self.cropped_image.shape[2]

        self.cache = {}

        self.parent_img_patch = parent_img_patch

        self.horizontal_center = (self.left + self.right) / 2
        self.vertical_center = (self.lower + self.upper) / 2

        if self.cropped_image.shape[1] == 0 or self.cropped_image.shape[2] == 0:
            raise Exception("ImagePatch has no area")

        self.possible_options = load_json('./useful_lists/possible_options.json')

        self.queues = queues

    def forward(self, **kwargs): # kwargs should have process name
        return forward(queues=self.queues, **kwargs)

    @property
    def original_image(self):
        if self.parent_img_patch is None:
            return self.cropped_image
        else:
            return self.parent_img_patch.original_image

    def find(self, object_name: str) -> list[ImagePatch]:
        """Returns a list of ImagePatch objects matching object_name contained in the crop if any are found.
        Otherwise, returns an empty list.
        Parameters
        ----------
        object_name : str
            the name of the object to be found

        Returns
        -------
        List[ImagePatch]
            a list of ImagePatch objects matching object_name contained in the crop
        """
        all_object_coordinates = self.forward(process_name = image_patch_api_config.find.use_process,
                                                  image = self.cropped_image,
                                                  label = object_name)
        # if object_name in ["object", "objects"]:
        #     all_object_coordinates = self.forward(process_name = 'maskrcnn_find',
        #                                           image = self.cropped_image)[0]
        # else:

        #     if object_name == 'person':
        #         object_name = 'people'  # GLIP does better at people than person

        #     all_object_coordinates = self.forward(process_name = 'glip_find',
        #                                           image = self.cropped_image,
        #                                           obj = object_name)
        # if len(all_object_coordinates) == 0:
        #     return []

        threshold = image_patch_api_config.find.ratio_box_area_to_image_area
        if threshold > 0:
            area_im = self.width * self.height
            all_areas = torch.tensor([(coord[2]-coord[0]) * (coord[3]-coord[1]) / area_im
                                      for coord in all_object_coordinates])
            mask = all_areas > threshold
            # if not mask.any():
            #     mask = all_areas == all_areas.max()  # At least return one element
            all_object_coordinates = all_object_coordinates[mask]

        return [self.crop(*coordinates) for coordinates in all_object_coordinates]

    def exists(self, object_name) -> bool:
        """Returns True if the object specified by object_name is found in the image, and False otherwise.
        Parameters
        -------
        object_name : str
            A string describing the name of the object to be found in the image.
        """
        if object_name.isdigit() or object_name.lower().startswith("number"):
            object_name = object_name.lower().replace("number", "").strip()

            object_name = w2n.word_to_num(object_name)
            answer = self.simple_query("What number is written in the image (in digits)?")
            return w2n.word_to_num(answer) == object_name

        patches = self.find(object_name)

        filtered_patches = []
        for patch in patches:
            if "yes" in patch.simple_query(f"Is this a {object_name}?"):
                filtered_patches.append(patch)
        return len(filtered_patches) > 0

    def verify_property(self, object_name: str, attribute: str) -> bool:
        """Returns True if the object possesses the property, and False otherwise.
        Differs from 'exists' in that it presupposes the existence of the object specified by object_name, instead
        checking whether the object possesses the property.
        Parameters
        -------
        object_name : str
            A string describing the name of the object to be found in the image.
        attribute : str
            A string describing the property to be checked.
        """
        name = f"{attribute} {object_name}"
        process_name = image_patch_api_config.verify_property.use_process
        negative_categories = [f"{att} {object_name}" for att in self.possible_options['attributes']]
        
        # not using
        # if process_name == 'clip_verify_property':
            # score = self.forward(process_name = process_name,
            #                      image = self.cropped_image,
            #                      category = name,
            #                      negative_categories=negative_categories)
            # thresh = models_config.clip.processes.clip_verify_property.threshold
        # if process_name == 'tcl_verify_property':
        #     score = self.forward(, self.cropped_image, name, task='score')
        #     thresh = models_config.clip.processes.clip_verify_property.threshold
        if process_name == 'xvlm_verify_property':
            binary_score = negative_categories is not None 
            score = self.forward(process_name = process_name,
                                 image = self.cropped_image,
                                 text = name,
                                 binary_score=binary_score,
                                 negative_categories=negative_categories)
            score = score.item()
            thresh = models_config.xvlm.processes.xvlm_verify_property.threshold
        
        return score > thresh

    def best_text_match(self, option_list: list[str] = None, prefix: str = None) -> str:
        """Returns the string that best matches the image.
        Parameters
        -------
        option_list : str
            A list with the names of the different options
        prefix : str
            A string with the prefixes to append to the options
        """
        option_list_to_use = option_list
        if prefix is not None:
            option_list_to_use = [prefix + " " + option for option in option_list]

        process_name = image_patch_api_config.best_text_match.use_process
        image = self.cropped_image
        text = option_list_to_use
        # if process_name in ('clip', 'tcl'):
        #     selected = self.forward(process_name, image, text, task='classify')
        if process_name == 'xvlm_best_text_match':
            res = self.forward(process_name = process_name,
                               image = image,
                               text = text,
                               binary_score = False)
            res = res.argmax().item()
            selected = res
        else:
            raise NotImplementedError

        return option_list[selected]

    def simple_query(self, question: str):
        """Returns the answer to a basic question asked about the image. If no question is provided, returns the answer
        to "What is this?". The questions are about basic perception, and are not meant to be used for complex reasoning
        or external knowledge.
        Parameters
        -------
        question : str
            A string describing the question to be asked.
        """
        return self.forward(
            process_name = 'blip_qa',
            image = self.cropped_image,
            question = question
        )

    ### OBSOLETE - NOT USED
    # def compute_depth(self):
    #     """Returns the median depth of the image crop
    #     Parameters
    #     ----------
    #     Returns
    #     -------
    #     float
    #         the median depth of the image crop
    #     """
    #     original_image = self.original_image
    #     depth_map = self.forward(
    #         process_name = 'midas_depth',
    #         image = original_image
    #     )
    #     depth_map = depth_map[original_image.shape[1]-self.upper:original_image.shape[1]-self.lower,
    #                           self.left:self.right]
    #     return depth_map.median()  # Ideally some kind of mode, but median is good enough for now

    def crop(self, left: int, lower: int, right: int, upper: int) -> ImagePatch:
        """Returns a new ImagePatch containing a crop of the original image at the given coordinates.
        Parameters
        ----------
        left : int
            the position of the left border of the crop's bounding box in the original image
        lower : int
            the position of the bottom border of the crop's bounding box in the original image
        right : int
            the position of the right border of the crop's bounding box in the original image
        upper : int
            the position of the top border of the crop's bounding box in the original image

        Returns
        -------
        ImagePatch
            a new ImagePatch containing a crop of the original image at the given coordinates
        """
        # make all inputs ints
        left = int(left)
        lower = int(lower)
        right = int(right)
        upper = int(upper)

        if image_patch_api_config.crop.crop_larger_margin:
            left = max(0, left - 10)
            lower = max(0, lower - 10)
            right = min(self.width, right + 10)
            upper = min(self.height, upper + 10)

        return ImagePatch(self.cropped_image, left, lower, right, upper, self.left, self.lower,
                          parent_img_patch=self, queues=self.queues, )

    def overlaps_with(self, left, lower, right, upper):
        """Returns True if a crop with the given coordinates overlaps with this one,
        else False.
        Parameters
        ----------
        left : int
            the left border of the crop to be checked
        lower : int
            the lower border of the crop to be checked
        right : int
            the right border of the crop to be checked
        upper : int
            the upper border of the crop to be checked

        Returns
        -------
        bool
            True if a crop with the given coordinates overlaps with this one, else False
        """
        return self.left <= right and self.right >= left and self.lower <= upper and self.upper >= lower

    def llm_query(self, question: str, long_answer: bool = True) -> str:
        return llm_query(question, None, long_answer)

    def print_image(self, size: tuple[int, int] = None):
        show_single_image(self.cropped_image, size)

    def __repr__(self):
        return "ImagePatch({}, {}, {}, {})".format(self.left, self.lower, self.right, self.upper)

def distance(patch_a: Union[ImagePatch, float], patch_b: Union[ImagePatch, float]) -> float:
    """
    Returns the distance between the edges of two ImagePatches, or between two floats.
    If the patches overlap, it returns a negative distance corresponding to the negative intersection over union.
    """

    if isinstance(patch_a, ImagePatch) and isinstance(patch_b, ImagePatch):
        a_min = np.array([patch_a.left, patch_a.lower])
        a_max = np.array([patch_a.right, patch_a.upper])
        b_min = np.array([patch_b.left, patch_b.lower])
        b_max = np.array([patch_b.right, patch_b.upper])

        u = np.maximum(0, a_min - b_max)
        v = np.maximum(0, b_min - a_max)

        dist = np.sqrt((u ** 2).sum() + (v ** 2).sum())

        if dist == 0:
            box_a = torch.tensor([patch_a.left, patch_a.lower, patch_a.right, patch_a.upper])[None]
            box_b = torch.tensor([patch_b.left, patch_b.lower, patch_b.right, patch_b.upper])[None]
            dist = - box_iou(box_a, box_b).item()

    else:
        dist = abs(patch_a - patch_b)

    return dist


def bool_to_yesno(bool_answer: bool) -> str:
    """Returns a yes/no answer to a question based on the boolean value of bool_answer.
    Parameters
    ----------
    bool_answer : bool
        a boolean value

    Returns
    -------
    str
        a yes/no answer to a question based on the boolean value of bool_answer
    """
    return "yes" if bool_answer else "no"

# OBSOLETE
# def process_guesses(prompt, guess1=None, guess2=None, output_queue_to_use=None):
#     return forward(
#         process_name='gpt_guess',
#         prompt=[prompt, guess1, guess2],
#         output_queue_to_use=output_queue_to_use
#     )


def coerce_to_numeric(string, no_string=False):
    """
    This function takes a string as input and returns a numeric value after removing any non-numeric characters.
    If the input string contains a range (e.g. "10-15"), it returns the first value in the range.
    # TODO: Cases like '25to26' return 2526, which is not correct.
    """
    if any(month in string.lower() for month in ['january', 'february', 'march', 'april', 'may', 'june', 'july',
                                                 'august', 'september', 'october', 'november', 'december']):
        try:
            return dateparser.parse(string).timestamp().year
        except:  # Parse Error
            pass

    try:
        # If it is a word number (e.g. 'zero')
        numeric = w2n.word_to_num(string)
        return numeric
    except ValueError:
        pass

    # Remove any non-numeric characters except the decimal point and the negative sign
    string_re = re.sub("[^0-9\.\-]", "", string)

    if string_re.startswith('-'):
        string_re = '&' + string_re[1:]

    # Check if the string includes a range
    if "-" in string_re:
        # Split the string into parts based on the dash character
        parts = string_re.split("-")
        return coerce_to_numeric(parts[0].replace('&', '-'))
    else:
        string_re = string_re.replace('&', '-')

    try:
        # Convert the string to a float or int depending on whether it has a decimal point
        if "." in string_re:
            numeric = float(string_re)
        else:
            numeric = int(string_re)
    except:
        if no_string:
            raise ValueError
        # No numeric values. Return input
        return string
    return numeric

def best_image_match(list_patches: list[ImagePatch], content: List[str], return_index: bool = False) -> \
        Union[ImagePatch, None]:
    """Returns the patch most likely to contain the content.
    Parameters
    ----------
    list_patches : List[ImagePatch]
    content : List[str]
        the object of interest
    return_index : bool
        if True, returns the index of the patch most likely to contain the object

    Returns
    -------
    int
        Patch most likely to contain the object
    """
    if len(list_patches) == 0:
        return None

    scores = []
    for cont in content:
        res = list_patches[0].forward(
            process_name = 'clip_best_image_match',
            images = [p.cropped_image for p in list_patches],
            prompt = cont,
            return_scores=True
        )
        scores.append(res)
    scores = torch.stack(scores).mean(dim=0)

    # (we're voting; content may be different descriptions of the same intended thing)
    scores = scores.argmax().item()  # Argmax over all image patches

    if return_index:
        return scores
    return list_patches[scores]

