import os
from PIL import Image
from torch.utils.data import Dataset
import json
import random
from tqdm import tqdm
import numpy as np
import torch
from typing import Tuple, List

from faceutils.inversions import accelerated_invert, ddim_invert, cached_inversion
from faceutils.utils import image2latent, image2tensor

from faceutils.constants import AGE_GENDER_RACE_MAP


def generate_prompt(gender: str='', age: str='', race: str=''):
    """generates prompt according to attribute strings"""
    assert gender or (age and gender and race) or not (age or gender or race)
    if not gender:
        return "A face"
    if not age:
        return f"The face of a {gender}"
    gender_possessive = "his" if gender == "man" else "her"
    return f"The face of a racially {race} {gender} in {gender_possessive} {age}"


class ImageDataset(Dataset):
    def __init__(self, image_dir:str, res:int=0, num_imgs:int=0):
        self.image_dir = image_dir
        self.image_filenames = sorted(os.listdir(image_dir))
        if num_imgs:
            self.image_filenames = self.image_filenames[:num_imgs]
        self.res = res

    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx) -> torch.Tensor:
        """Returns torch.Tensor in the shape (C x H x W) scaled to [-1.0, 1.0]"""
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path)
        if self.res:
            image = image.resize((self.res, self.res))
        image = np.array(image)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 127.5 - 1
        return image
    

class FaceIDDataset(Dataset):
    def __init__(self, image_dir, file_list=[], res=256):
        self.image_dir = image_dir
        self.image_filenames = file_list if file_list else os.listdir(image_dir)
        self.res = res
        self._identity = image_dir.split("/")[-1]

    @property
    def identity(self) -> str:
        return self._identity 

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx) -> torch.Tensor:
        """Returns torch.Tensor in the shape (C x H x W) scaled to [-1.0, 1.0]"""
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).resize((self.res, self.res))
        image = np.array(image)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 127.5 - 1

        return image


class IDPromptDataset(FaceIDDataset):
    def __init__(self, image_dir, attribute_path, id_attribute_path, prompt_type, file_list=[], res=256):
        assert prompt_type in ["gender", "age_gender_race", "face"]
        super().__init__(image_dir, file_list, res)
        with open(attribute_path, "r") as file:
            self.attributes = json.load(file)
        with open(id_attribute_path, "r") as file:
            self.id_attributes = json.load(file)
        self.prompt_type = prompt_type
        self._gender = self.id_attributes[self.identity]["gender"]
        self._race = self.id_attributes[self.identity]["race"]
        self.prompts = []
        for img_name in self.image_filenames:
            if self.prompt_type == "face":
                prompt_str = "face"
            elif self.prompt_type == "gender":
                prompt_str = generate_prompt(self.gender)
            else:
                img_no = os.path.splitext(img_name)[0]
                age = self.attributes[img_no]["age"]
                age_str = AGE_GENDER_RACE_MAP["age"][age]
                prompt_str = generate_prompt(self.gender, age_str, self.race)
            self.prompts.append(prompt_str)

    @property
    def gender(self) -> str:
        gender_str = AGE_GENDER_RACE_MAP["gender"][self._gender]
        return gender_str
    
    @property
    def race(self) -> str:
        race_str = AGE_GENDER_RACE_MAP["race"][self._race]
        return race_str
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, str]:
        """Returns torch.Tensor in the shape (C x H x W) scaled to [-1.0, 1.0] and a prompt str"""
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).resize((self.res, self.res))
        image = np.array(image)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 127.5 - 1

        return image, self.prompts[idx]


class IDLatentPromptDataset(IDPromptDataset):
    def __init__(self, model, config, image_dir, file_list=[], inversion_fn=accelerated_invert):
        attribute_path = config.dataset.attribute_path
        id_attribute_path = config.dataset.id_attribute_path
        prompt_type = config.dataset.prompt_type
        res = config.dataset.res
        super().__init__(image_dir, attribute_path, id_attribute_path, prompt_type, file_list, res)
        num_images = len(self.image_filenames)
        self.device = model.device
        self.inversion_fn = cached_inversion(inversion_fn, config, model)
        self.latents = []
        print(f"{self.identity} image inversions...")
        for idx in tqdm(range(num_images), bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}'):
            img_name = self.image_filenames[idx]
            img_path = os.path.join(self.image_dir, img_name)
            latent = self.inversion_fn(
                img_path = img_path,
                prompt = self.prompts[idx],
                guidance_scale=config.diffusion.guidance_scale, 
                num_inference_steps=config.diffusion.diffusion_steps,
                num_fixed_point_iters=config.diffusion.num_fixed_point_iters,
                start_step=config.diffusion.start_step
            )
            self.latents.append(latent)
        self.latents = torch.cat(self.latents).cpu()
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Returns torch.Tensor in the shape (C x H x W) scaled to [-1.0, 1.0] and a prompt str"""
        image, prompt =  super().__getitem__(idx)
        latent = self.latents[idx]
        return image, latent, prompt


class GroupLatentPromptDataset(Dataset):
    def __init__(self, dataset_list: List[IDLatentPromptDataset], num_from_each_set: int, slice_idx: int):
        self.dataset_list = dataset_list
        self.num_from_each = num_from_each_set
        self.slice_idx = slice_idx

        sub_dataset_len = len(dataset_list[0])
        assert sub_dataset_len % num_from_each_set == 0
        assert slice_idx < sub_dataset_len // num_from_each_set # slice_idx < num_slices

        self.num_dataset = len(dataset_list)
        self.identities = []
        for dataset in dataset_list:
            self.identities.append(dataset.identity)
        self.identity = '_'.join(self.identities) + f'_slice{slice_idx}'
        self.image_filenames = []
        for idx in range(len(self)):
            self.image_filenames.append(self._get_img_file(idx))
    
    def __len__(self):
        return self.num_dataset * self.num_from_each
    
    def _get_img_file(self, idx):
        dataset_i = idx // self.num_from_each
        data_i = self.slice_idx * self.num_from_each + idx % self.num_from_each
        return self.dataset_list[dataset_i].image_filenames[data_i]

    def __getitem__(self, idx):
        dataset_i = idx // self.num_from_each
        data_i = self.slice_idx * self.num_from_each + idx % self.num_from_each
        return self.dataset_list[dataset_i][data_i]


class FamilyLatentDataset(Dataset):
    def __init__(self, model, config, image_dir, file_list=[], inversion_fn=accelerated_invert):
        """file_list: a list of [img_id, img_filename] pairs"""
        with open(config.dataset.id_attribute_path, 'r') as f:
            id2gender = json.load(f)
        self.res = config.dataset.res
        self.image_filenames = [img_filename for img_id, img_filename in file_list]
        self.image_dir = image_dir

        self.prompts = []
        self.img_ids = []
        for img_id, _ in file_list:
            gender = id2gender[img_id]
            self.prompts.append(generate_prompt(gender=gender))
            if img_id not in self.img_ids:
                self.img_ids.append(img_id)

        num_images = len(self.image_filenames)
        self.device = model.device
        self.inversion_fn = cached_inversion(inversion_fn, config, model)

        self.device = model.device
        self.inversion_fn = cached_inversion(inversion_fn, config, model)
        self.latents = []

        print(f"{self.img_ids} image inversions...")
        for idx in tqdm(range(num_images), bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}'):
            img_name = self.image_filenames[idx]
            img_path = os.path.join(self.image_dir, img_name)
            latent = self.inversion_fn(
                img_path = img_path,
                prompt = self.prompts[idx],
                guidance_scale=config.diffusion.guidance_scale, 
                num_inference_steps=config.diffusion.diffusion_steps,
                num_fixed_point_iters=config.diffusion.num_fixed_point_iters,
                start_step=config.diffusion.start_step
            )
            self.latents.append(latent)
        self.latents = torch.cat(self.latents).cpu()

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, str]:
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).resize((self.res, self.res))
        image = np.array(image)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 127.5 - 1

        prompt = self.prompt[idx]
        latent = self.latents[idx]

        return image, latent, prompt