from PIL import Image
import argparse
from functools import partial
from tqdm import tqdm
from typing import Dict, Callable
import os
import json

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from faceutils import defense
from faceutils.utils import ModelEnsemble, load_test_models, image2tensor
from faceutils.datasets import ImageDataset
from faceutils.datetime_logger import DatetimeLogger
from faceutils.constants import THRES_DICT


def robustness_test(
    img_dir: str, 
    target_embs: Dict[str, torch.Tensor], 
    fr_models: ModelEnsemble, 
    test_model_name: str, 
    defense_fn: Callable, 
    batch_size:int=8,
    save_defense: bool=False,
) -> float:

    testset = ImageDataset(img_dir)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    device = list(target_embs.values())[0].device
    
    asc = 0
    with torch.no_grad():
        for batch in tqdm(testloader, bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}'):
            batch: torch.Tensor
            batch = batch.to(device)
            batch_emb_defense = fr_models(batch, [test_model_name], defense_fn, save_defense)
            simi = F.cosine_similarity(target_embs[test_model_name], batch_emb_defense[test_model_name])
            thres = THRES_DICT[test_model_name][1]
            success = simi > thres
            asc += success.sum().item()
    asr = asc / len(testset)
    return asr


def get_defense_fn(defense_method: str, args) -> Callable:
    if defense_method == "undefended":
        defense_fn = None
    elif defense_method == "median_blur":
        defense_fn = partial(defense.median_blur, kernel_size=args.kernel_size)
    elif defense_method == "feature_squeezing":
        defense_fn = partial(defense.feature_squeezing, bit_depth=args.bit_depth)
    elif defense_method == "gaussian_blur":
        defense_fn = partial(defense.gaussian_blur, kernel_size=args.kernel_size)
    elif defense_method == "jpeg":
        defense_fn = partial(defense.jpeg_defense, quality=args.jpeg_quality)
    else:
        raise ValueError("Invalid defense method.")
    return defense_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--img_root", default="../datasets/all_gen_imgs/", type=str)
    parser.add_argument("--test_img", default="../datasets/celebahq_all_images/test/047073.jpg", type=str)
    parser.add_argument("--fr_model_names", nargs='+', default=['mobile_face', 'ir152', 'irse50', 'facenet'])
    # parser.add_argument("--test_model_name", default="facenet", type=str)
    parser.add_argument("--fr_model_dir", default="./assets/victim_models", type=str)
    parser.add_argument("--defenses", nargs='+', default=["undefended", "median_blur", "feature_squeezing", "gaussian_blur", "jpeg"])
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--bit_depth", default=3, type=int)
    parser.add_argument("--kernel_size", default=5, type=int, help="The kernel size for Gaussian blur and median blur. Must be and odd integer.")
    parser.add_argument("--jpeg_quality", default=5, type=int)
    args = parser.parse_args()

    logger = DatetimeLogger("../logs", "robustness_test")
    logger.log(f"bit_depth: {args.bit_depth}")
    logger.log(f"kernel_size: {args.kernel_size}")
    logger.log(f"jpeg_quality: {args.jpeg_quality}")
    logger.log()

    img_root = args.img_root
    defenses = args.defenses
    test_model_names = args.fr_model_names
    attack_methods = os.listdir(img_root)

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    fr_models = load_test_models(args.fr_model_names, args.fr_model_dir, device)
    fr_models = ModelEnsemble(fr_models, device)

    target_img = Image.open(args.test_img)
    target_img = image2tensor(target_img, device)
    target_embs = fr_models(target_img)

    asr_dict = {am: {tmn: {defense_method: 0 for defense_method in defenses} for tmn in test_model_names} for am in attack_methods}

    for am in attack_methods:
        logger.log(f"{'    '*0}Testing {am}")
        for tmn in test_model_names:
            logger.log(f"{'    '*1}On target model {tmn}:")
            tmn_dir = os.path.join(img_root, am, tmn)
            for defense_method in defenses:
                logger.log(f"{'    '*2}Against {defense_method} defense.")
                defense_fn = get_defense_fn(defense_method, args)
                save_defense = False
                # save_defense = am != "amt-gan" and defense_method == "median_blur"
                if am != "amt-gan":
                    img_dir = tmn_dir
                    defended_asr = robustness_test(img_dir, target_embs, fr_models, tmn, defense_fn, args.batch_size, save_defense=save_defense)
                    logger.log(f"{'    '*3}ASR: {defended_asr}")
                else:
                    refs = sorted(os.listdir(tmn_dir))
                    defended_asr = 0
                    for ref in refs:
                        img_dir = os.path.join(tmn_dir, ref)
                        defended_asr_ref = robustness_test(img_dir, target_embs, fr_models, tmn, defense_fn, args.batch_size)
                        defended_asr += defended_asr_ref
                    defended_asr = defended_asr / len(refs)
                    logger.log(f"{'    '*3}avaerage ASR: {defended_asr}")
                
                asr_dict[am][tmn][defense_method] = 100 * defended_asr
    
    with open(os.path.join(logger.log_dir, "defended_asrs.json"), 'w') as f:
        json.dump(asr_dict, f, indent=4)
                        
