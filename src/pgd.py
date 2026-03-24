import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image

import argparse
import random
import os
from typing import Dict

from faceutils.utils import ModelEnsemble, load_test_models, tensor2image
from faceutils.datetime_logger import DatetimeLogger
from faceutils.datasets import ImageDataset
from faceutils.constants import THRES_DICT
from faceutils.timer import Timer

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', default="../datasets/celebahq_all_images/imgs", type=str)
parser.add_argument('--target_path', default="../datasets/celebahq_all_images/target/085807.jpg", type=str)
parser.add_argument('--test_path', default="../datasets/celebahq_all_images/test/047073.jpg", type=str)
parser.add_argument('--log_dir', default="../logs", type=str)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--res', default=160, type=int)
parser.add_argument('--num_imgs', default=800, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--radius', default=0.05, type=float)
parser.add_argument('--lr', default=0.005, type=float)
parser.add_argument('--momentum', default=0.0, type=float)
parser.add_argument('--iters', default=80, type=int)
parser.add_argument('--warm_up_iters', default=8, type=int)
parser.add_argument('--norm_type', default='linf', type=str, choices=['linf', 'l2'])
parser.add_argument('--surrogate_model_names', nargs='+', default=['mobile_face', 'ir152', 'irse50'])
parser.add_argument('--test_model_names', nargs='+', default=['mobile_face', 'ir152', 'irse50', 'facenet'])
parser.add_argument('--victim_model_dir', default="./fr_models", type=str)
parser.add_argument('--do_attack', default=1, type=int)


def seed_torch(seed=42):
    """For reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def projection(x: torch.Tensor, eps: float, norm_type: str="linf"):
    assert norm_type in ["linf", "l2"]
    if norm_type == "linf":
        return torch.clamp(x, min=-eps, max=eps)
    else:
        flat = x.view(x.size(0), -1)
        l2_norm = flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
        scale = eps / l2_norm
        scale = torch.minimum(scale, torch.ones_like(scale))
        return x * scale.view(-1, 1, 1, 1)


class AdvOptimizer():
    def __init__(
        self, 
        variable: torch.Tensor, 
        lr: float=0.01, 
        radius: float=0.1, 
        norm_type: str="linf", 
        momentum: float=0.0,
        warm_up_iters: int=8,
    ):
        assert norm_type in ["linf", "l2"]
        self.variable = variable
        self.lr = lr
        self.radius = radius
        self.norm_type = norm_type
        self.momentum = momentum
        self.warm_up_iters = warm_up_iters
        self.num_iters = 0
        self.buffer = None

    @torch.no_grad()
    def projection(self):
        if self.norm_type == "linf":
            self.variable.clamp_(min=-self.radius, max=self.radius)
        else:
            flat = self.variable.view(self.variable.size(0), -1)
            l2_norm = flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
            scale = self.radius / l2_norm
            scale = torch.minimum(scale, torch.ones_like(scale))
            self.variable.mul_(scale.view(-1, 1, 1, 1))
    
    @torch.no_grad()
    def normalize_grad(self, x: torch.Tensor):
        flat = x.view(x.size(0), -1)
        if self.norm_type == "linf":
            norm = flat.norm(p=1, dim=1, keepdim=True).clamp(min=1e-12)
            return x / norm.view(-1, 1, 1, 1)
        else:
            norm = flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
            return x / norm.view(-1, 1, 1, 1)
    
    @torch.no_grad()
    def update(self, grad: torch.Tensor):
        if self.num_iters < self.warm_up_iters:
            lr = self.lr * (self.num_iters + 1) / self.warm_up_iters
        else:
            lr = self.lr
        if not self.momentum > 0:
            if self.norm_type == "linf":
                update_step = lr * torch.sign(grad)
            else:
                update_step = lr * grad
        else:
            if self.buffer is None:
                self.buffer = self.normalize_grad(grad)
            else:
                self.buffer = self.momentum * self.buffer + self.normalize_grad(grad)
            update_step = self.buffer
        self.variable.sub_(update_step)
        self.projection()
        self.num_iters += 1
    
    @torch.no_grad()
    def __call__(self, grad: torch.Tensor):
        assert grad.shape == self.variable.shape
        self.update(grad)


def cos_dist_loss(src_features: Dict[str, torch.Tensor], tgt_feature: Dict[str, torch.Tensor]) -> torch.Tensor: 
    """
    Calculates cos_dist ,i.e. (1 - cos_simi), between source and target features 
    produced by (possibly) multiple target models.
    That is, both source_feature and target_feature are outputs of VictimModels.
    Items in both input dicts are of the format {model_name: batch_embedding}.
    """
    cos_dist_list = []
    for model_name in src_features.keys():
        cos_dist = 1 - F.cosine_similarity(src_features[model_name], tgt_feature[model_name])
        cos_dist_list.append(cos_dist)
    cos_loss = torch.mean(torch.stack(cos_dist_list), dim=0) # take mean over models
    cos_loss = torch.sum(cos_loss) # sum over batch
    return cos_loss


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    seed_torch(args.seed)

    for mn in args.test_model_names:
        if mn not in args.surrogate_model_names:
            attacked_model = mn
    if args.iters == 1:
        attack_algo = "fgsm"
    elif args.momentum < 1e-4:
        attack_algo = 'pgd'
    else:
        attack_algo = 'mifgsm'

    logger = DatetimeLogger(args.log_dir, f"{attack_algo}_{attacked_model}")
    logger.log()
    logger.log(f"res: {args.res}")
    logger.log(f"num_imgs: {args.num_imgs}")
    logger.log(f"batch_size: {args.batch_size}")
    logger.log(f"radius: {args.radius}")
    logger.log(f"lr: {args.lr}")
    logger.log(f"iters: {args.iters}")
    logger.log(f"warm_up_iters: {args.warm_up_iters}")
    logger.log(f"norm_type: {args.norm_type}")
    logger.log(f"surrogate_model_names: {args.surrogate_model_names}")
    logger.log(f"test_model_names: {args.test_model_names}")
    logger.log()

    im_save_dir = os.path.join(logger.log_dir, "adv_imgs")
    os.makedirs(im_save_dir, exist_ok=True)

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    surrogate_model_names = args.surrogate_model_names
    test_model_names = args.test_model_names
    victim_model_dir = args.victim_model_dir

    # load fr models
    print("Loading FR models...")
    surrogate_models = load_test_models(surrogate_model_names, victim_model_dir, device)
    surrogate_models = ModelEnsemble(surrogate_models, device)
    surrogate_models.requires_grad_(False)
    test_models = load_test_models(test_model_names, victim_model_dir, device)
    test_models = ModelEnsemble(test_models, device)
    test_models.requires_grad_(False)

    # create dataset
    print("Creating dataset and dataloader...")
    dataset = ImageDataset(args.image_dir, args.res, args.num_imgs)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Process target image
    target_img = Image.open(args.target_path).resize((args.res, args.res))
    target_img = np.array(target_img)
    target_img = torch.from_numpy(target_img).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1
    target_img = target_img.to(device)
    target_emb = surrogate_models(target_img)
    target_test_emb = test_models(target_img)

    # Process test image
    test_img = Image.open(args.test_path).resize((args.res, args.res))
    test_img = np.array(test_img)
    test_img = torch.from_numpy(test_img).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1
    test_img = test_img.to(device)
    test_emb = test_models(test_img)

    ascs = {model_name: 0 for model_name in test_model_names}

    timer = Timer()
    timer.tic()
    # Training loop
    for batch_no, batch_imgs in enumerate(dataloader):
        logger.log(f"Training batch no {batch_no+1} out of {args.num_imgs // args.batch_size}")
        logger.log(f"Images in batch: {dataset.image_filenames[batch_no*args.batch_size: batch_no*args.batch_size+args.batch_size]}")
        batch_imgs = batch_imgs.to(device)
        delta = torch.zeros_like(batch_imgs).to(device)
        delta.requires_grad_(True)

        optimizer = AdvOptimizer(
            variable=delta,
            lr=args.lr,
            radius=args.radius,
            norm_type=args.norm_type,
            momentum=args.momentum,
            warm_up_iters=args.warm_up_iters,
        )

        # train delta
        if args.do_attack:
            for iter in tqdm(range(args.iters), bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}'):
                adv_imgs = batch_imgs + delta
                adv_embs = surrogate_models(adv_imgs)

                # if iter < args.warm_up_iters:
                #     lr = args.lr * (iter + 1) / args.warm_up_iters
                # else:
                #     lr = args.lr

                id_loss = cos_dist_loss(adv_embs, target_emb)
                grad = torch.autograd.grad(id_loss, delta)[0]

                optimizer(grad.detach())
        
        # eval delta
        with torch.inference_mode():
            adv_imgs = torch.clamp(batch_imgs + delta, min=-1, max=1)
            adv_test_emb = test_models(adv_imgs)
            for mn in test_model_names:

                simi = F.cosine_similarity(adv_test_emb[mn], test_emb[mn], dim=1)
                success = simi > THRES_DICT[mn][1]
                batch_asc = torch.sum(success).item()
                ascs[mn] += batch_asc
                logger.log(f"    Succeeded {batch_asc} out of {args.batch_size} times on {mn}")
        
        # save adv images
        if args.do_attack:
            adv_imgs_pil = tensor2image(adv_imgs)
            for iter, adv_img in tqdm(enumerate(adv_imgs_pil)):
                idx_filename = batch_no * args.batch_size + iter
                orginal_file_name = dataset.image_filenames[idx_filename]
                img_no, _ = os.path.splitext(orginal_file_name)
                save_file_name = img_no + ".png"
                save_file_path = os.path.join(im_save_dir, save_file_name)
                # print(f"Saving adv images to {save_file_path}")
                adv_img.save(save_file_path)
    
    training_time = timer.toc()
    asrs = {mn: cnt / args.num_imgs for mn, cnt in ascs.items()}
    logger.log(f"ASR:")
    for mn, asr in asrs.items():
        logger.log(f"    {mn}: {asr*100:.3f}")
    peak_memory = torch.cuda.max_memory_allocated()
    logger.log(f"Peak memory usage: {peak_memory / (1024 ** 2):.2f} MB")
    logger.log(f"training time: {training_time}")

