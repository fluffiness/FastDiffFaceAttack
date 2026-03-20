import argparse
import os
from tqdm import tqdm

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from faceutils.datetime_logger import DatetimeLogger
from faceutils.utils import ModelEnsemble, load_test_models, SemSegCELoss
from faceutils.timer import Timer

from faceutils.utils import image2tensor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_img_dir', default="../logs/diffprot_gen_imgs/val/atkmodel-ir152_save_eps0.02_Tenc100_iter50_Tatk5_Tstart5_Tinf100_atk1_atkinf4_repeat1_skip0_lam0.0", type=str)
    parser.add_argument('--test_model_name', default="facenet", type=str)
    parser.add_argument('--fr_model_dir', default="./assets/victim_models", type=str)
    parser.add_argument('--test_file_path', default="../datasets/celebamaskhq_identity_dataset/test/047073.jpg", type=str)
    args = parser.parse_args()

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    # res=512
    THRES_DICT = {'ir152': (0.094632, 0.166788, 0.227922), 'irse50': (0.144840, 0.241045, 0.312703),
                    'facenet': (0.256587, 0.409131, 0.591191), 'mobile_face': (0.183635, 0.301611, 0.380878), 
                    'cosface': (0.144840, 0.241045, 0.312703), 'arcface': (0.144840, 0.241045, 0.312703)}

    test_models = load_test_models([args.test_model_name], args.fr_model_dir, device)
    test_models = ModelEnsemble(test_models, device)

    target_file_path = args.test_file_path
    target_img = Image.open(target_file_path)
    target_img = image2tensor(target_img, device)
    target_emb = test_models(target_img)

    refs = sorted(os.listdir(args.gen_img_dir))

    total_img_cnt = 0
    asc = 0
    for img_file in tqdm(os.listdir(args.gen_img_dir), bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}'):
        if "jpg" not in img_file and "png" not in img_file:
            continue
        img_path = os.path.join(args.gen_img_dir, img_file)
        img = Image.open(img_path)
        img = image2tensor(img, device)
        adv_emb = test_models(img)

        simi = F.cosine_similarity(target_emb[args.test_model_name], adv_emb[args.test_model_name])
        thres = THRES_DICT[args.test_model_name][1]

        if simi > thres:
            asc += 1
        total_img_cnt += 1
    
    print(f"ASR on {args.test_model_name}: {asc*100/total_img_cnt:.3f}")
