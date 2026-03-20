import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union, Dict
import numpy as np
import os
from PIL import Image
from diffusers import DiffusionPipeline
from torch.utils.data import DataLoader
import torchvision.transforms.functional as Fv
import random
import re
from tqdm import tqdm

from faceutils.inversions import accelerated_invert, cached_inversion
from faceutils.utils import image2tensor, tensor2image, image2latent, latent2image, embed_prompt, get_noise_pred, diffusion_step, sample
from faceutils.utils import ModelEnsemble, cos_dist_loss, find_word_in_sentence, neighborhood_struct_loss
from faceutils.similarity_metrics import calculate_ssim, calculate_lpips
from faceutils.timer import Timer, sec2hms

from assets.semseg.model import BiSeNet


from faceutils.optimizers import AdamW, PGD
from faceutils.attention_control_utils import register_attention_control, reset_attention_control, aggregate_attention
from faceutils.attention_control import StructureLossAttentionStore
from faceutils.datetime_logger import DatetimeLogger
from faceutils.datasets import IDLatentPromptDataset, GroupLatentPromptDataset

from config import Config


"""
    The latents are duplicated and organized as follows:
        For the attention losses: latents = torch.cat([latents, adv_latents])
        For cfg: latents = torch.cat([latents * 2])
    During the noise prediction process, the input to the unit is 4 sets of the latents
    The first 2 are for the unconditional prediction, and the last 2 are for conditional
                  embeddings       latents
               ┌──────────────┬──────────────┐
     batch 1   │     null     │    clean     │
               ├──────────────┼──────────────┤
     batch 2   │     null     │     adv      │
               ├──────────────┼──────────────┤
     batch 3   │     text     │    clean     │
               ├──────────────┼──────────────┤
     batch 4   │     text     │     adv      │
               └──────────────┴──────────────┘
    In the AttentionControl modules, batch 1 and 2 are not stored since we only care about 
    the cross-attention with text embeddings.
    For self-attentions control, we calculate the MSE-loss of the self-attention maps
    between batch 3 and 4
"""

THRES_DICT = {'ir152': (0.094632, 0.166788, 0.227922), 'irse50': (0.144840, 0.241045, 0.312703),
            'facenet': (0.256587, 0.409131, 0.591191), 'mobile_face': (0.183635, 0.301611, 0.380878), 
            'cosface': (0.144840, 0.241045, 0.312703), 'arcface': (0.144840, 0.241045, 0.312703)}

KEY_WORDS = ["face", "man", "woman", "his", "her", "race", "racially", 
                 "white", "black", "asian", "indian","ambiguous", 
                 "teens", "twenties", "thirties", "forties", "fifties", "sixties", "seventies"]


def reduce_att_map(att_map: torch.Tensor, prompt: str, words: List[str], tokenizer):

    sorted_words = find_word_in_sentence(words, prompt)
    prompt_words = prompt.split(' ')

    # check the number of torkens in each word
    n_tokens = [0] + [len(tokenizer.encode(word))-2 for word in prompt_words] # -2 accounts for the start and end tokens
    cmul_tokens = [0] * len(n_tokens)
    for i in range(1, len(n_tokens)):
        cmul_tokens[i] = cmul_tokens[i-1] + n_tokens[i]
    
    prev_end_idx = 0
    reduced_map_components = []
    for word in sorted_words:
        word_idx = prompt_words.index(word)
        start_idx = cmul_tokens[word_idx]
        end_idx = cmul_tokens[word_idx+1]
        word_map = att_map[:, :, :, start_idx: end_idx] # (B, H, W, word_emb_len)
        reduced_map_components.append(att_map[:, :, :, prev_end_idx: start_idx])
        reduced_map_components.append(word_map.sum(dim=-1, keepdim=True))
        prev_end_idx = end_idx
    reduced_map_components.append(att_map[:, :, :, end_idx:])
    reduced_map = torch.cat(reduced_map_components, dim=-1)

    return reduced_map


def targeted_cross_attention_loss(
    controller: StructureLossAttentionStore, 
    prompts: Union[str, List[str]],
    prompt_type: str,
    target_prompt: str,
    target_att_map: torch.Tensor,
    tokenizer,
    res: int,
    w_ends: bool=True,
    w_blur: bool=True,
    kernel_size: int=3,
    kl_3d: bool=False,
) -> torch.Tensor:
    assert prompt_type in ["gender", "age_gender_race", "face"]

    batch_size = len(prompts)

    before_attention_map = aggregate_attention(batch_size, controller, res//32, ("up", "down"), True, 0, is_cpu=False)
    after_attention_map = aggregate_attention(batch_size, controller, res//32, ("up", "down"), True, 1, is_cpu=False)

    target_emb_len = len(tokenizer.encode(target_prompt)) - 2 # -2 to discard the <|startoftext|> and <|endoftext|> tokens
    target_sot_map = target_att_map[:, :, :, 0:1]
    target_text_map = target_att_map[:, :, :, 1: target_emb_len+1]
    target_eot_map = torch.sum(target_att_map[:, :, :, target_emb_len+1:], dim=-1, keepdim=True)

    att_maps = after_attention_map
    if prompt_type in ["gender", "face"]:
        p = prompts[0]
        emb_len = len(tokenizer.encode(p)) - 2 # -2 to discard the <|startoftext|> and <|endoftext|> tokens
        sot_maps = att_maps[:, :, :, 0:1]
        text_maps = att_maps[:, :, :, 1: emb_len+1]
        eot_maps = torch.sum(att_maps[:, :, :, emb_len+1:], dim=-1, keepdim=True)
        if w_ends:
            att_maps = torch.cat([sot_maps, text_maps, eot_maps], dim=-1)
            target_att_map = torch.cat([target_sot_map, target_text_map, target_eot_map], dim=-1)
        else:
            att_maps = text_maps
            target_att_map = target_text_map
    else:
        target_text_map = reduce_att_map(target_text_map, target_prompt, KEY_WORDS, tokenizer)
        reduced_att_maps = []
        for i, p in enumerate(prompts):
            emb_len = len(tokenizer.encode(p)) - 2
            sot_map = att_maps[i:i+1, :, :, 0:1]
            eot_map = torch.sum(att_maps[i:i+1, :, :, emb_len+1:], dim=-1, keepdim=True)
            text_map = att_maps[i:i+1, :, :, 1: emb_len+1]
            text_map = reduce_att_map(text_map, p, KEY_WORDS, tokenizer)
            if w_ends:
                att_map = torch.cat([sot_map, text_map, eot_map], dim=-1)
                target_att_map = torch.cat([target_sot_map, target_text_map, target_eot_map], dim=-1)
            else:
                att_map = text_map
                target_att_map = target_text_map
            reduced_att_maps.append(att_map)
        att_maps = torch.cat(reduced_att_maps, dim=0)

    eps = 1e-8
    if kl_3d:
        B, H, W, L = att_maps.shape
        P = att_maps / (H * W)
        Q = target_att_map / (H * W)
        P = P.clamp(min=eps)
        Q = Q.clamp(min=eps)
        kl = F.kl_div(P.log(), Q, reduction='none')
        ca_loss = kl.sum(dim=(1, 2, 3)).mean(dim=0)
        return ca_loss

    P = att_maps / att_maps.sum(dim=-1, keepdim=True)
    Q = target_att_map / target_att_map.sum(dim=-1, keepdim=True)

    if w_blur:
        P = Fv.gaussian_blur(P.permute((0, 3, 1, 2)), kernel_size=kernel_size, sigma=1.0)
        P = P.permute((0, 2, 3, 1))
        Q = Fv.gaussian_blur(Q.permute((0, 3, 1, 2)), kernel_size=kernel_size, sigma=1.0)
        Q = Q.permute((0, 2, 3, 1))

    P = P.clamp(min=eps)
    Q = Q.clamp(min=eps)

    kl = F.kl_div(P.log(), Q, reduction='none')  # shape: (B, H, W, L)
    kl_per_pixel = kl.sum(dim=-1) # sum over L → shape (B, H, W)
    ca_loss = kl_per_pixel.mean()

    return ca_loss


def cross_attention_loss(
        
    controller: StructureLossAttentionStore, 
    prompts: List[str],
    prompt_type: str,
    tokenizer,
    res: int,
) -> torch.Tensor:
    """
    Processes the stored attention maps in stored in controller and calculates the attention losses.
    There are 3 types of cross attention maps, with resolutions of res//8, res//16, and res//32.
    To prevent memory overhead, only those of resolutions res//16 and res//32 are stored.
    Here we only consider the lower res variety, perhaps to save memory?
    the returned maps are of dimensions (B, H, W, L)
    Note that cross-attention maps are normalized across the L dimension.
    That is, each pixel contains distribution of importance over the tokens.
    Also, Note that <|startoftext|>, <|endoftext|> tokens also take up, often significant, attention.
    The trailing empty tokens after <|endoftext|> also has some residual attention.
    """
    assert prompt_type in ["gender", "age_gender_race", "face"]

    batch_size = len(prompts)

    before_attention_map = aggregate_attention(batch_size, controller, res//32, ("up", "down"), True, 0, is_cpu=False)
    after_attention_map = aggregate_attention(batch_size, controller, res//32, ("up", "down"), True, 1, is_cpu=False)
    

    att_maps = after_attention_map
    if prompt_type in ["gender", "face"]:
        p = prompts[0]
        emb_len = len(tokenizer.encode(p)) - 2 # discard the <|startoftext|> and <|endoftext|> tokens
        att_maps = att_maps[:, :, :, 1: emb_len+1]
        # ca_loss = att_maps.var(dim=-1).mean() # variance across token distribution, then average across space and batch
        ca_loss = att_maps.var(dim=(1, 2, 3)).mean() # variance across space and token, then average across batch
    else:
        # ca_loss = 0.0
        # for i, p in enumerate(prompts):
        #     embedding_len = len(tokenizer.encode(p)) - 2
        #     att_map = att_maps[i, :, :, 1: embedding_len + 1] # (H, W, L)
        #     ca_loss += att_map.var()
        # ca_loss = ca_loss / batch_size
        reduced_att_maps = []
        for i, p in enumerate(prompts):
            emb_len = len(tokenizer.encode(p)) - 2
            sot_map = att_maps[i:i+1, :, :, 0:1]
            text_map = att_maps[i:i+1, :, :, 1: emb_len+1]
            text_map = reduce_att_map(text_map, p, KEY_WORDS, tokenizer)
            eot_map = torch.sum(att_maps[i:i+1, :, :, emb_len+1:], dim=-1, keepdim=True)
            reduced_att_maps.append(text_map)
        att_maps = torch.cat(reduced_att_maps, dim=0)
        ca_loss = att_maps.var(dim=(1, 2, 3)).mean()

    return ca_loss


@torch.enable_grad()
def identity_attack_memory_efficient( # for age_gender_race prompts
    model: DiffusionPipeline,
    trainset: Union[IDLatentPromptDataset, GroupLatentPromptDataset],
    target_data: dict,
    test_img: torch.Tensor,
    victim_models: ModelEnsemble,
    test_models: ModelEnsemble,
    parse_model: BiSeNet,
    log_ckpt_dir: str,
    config: Config,
    logger: DatetimeLogger,
) -> torch.Tensor:
    device = model.device
    num_inference_steps = config.diffusion.diffusion_steps
    start_step = config.diffusion.start_step
    guidance_scale = config.diffusion.guidance_scale
    do_classifier_free_guidance = (guidance_scale > 0.0)
    # clip_min = -1e6
    # clip_max = 1e6

    res = config.dataset.res
    prompt_type = config.dataset.prompt_type

    batch_size = config.training.batch_size
    lr = config.training.lr
    use_adaptive_lr = config.training.use_adaptive_lr
    pgd_radius = config.training.pgd_radius
    weight_decay = config.training.weight_decay
    alpha = config.training.attack_loss_weight
    beta = config.training.cross_attn_loss_weight
    gamma = config.training.self_attn_loss_weight
    n_epoch = config.training.total_update_steps // (len(trainset) // batch_size)
    lam = config.training.semseg_lam
    eta = config.training.pairwise_loss_weight
    zeta = config.training.neighborhood_loss_weight
    pairwise_loss_sign = config.training.pairwise_loss_sign
    ca_loss_is_targeted = config.training.ca_loss_is_targeted
    ca_loss_w_blur = config.training.ca_loss_w_blur
    ca_loss_kernel_size = config.training.ca_gaussian_kernel_size
    ca_loss_w_ends = config.training.ca_loss_is_w_ends
    kl_3d = config.training.kl_3d

    if lam > 0.0:
        from faceutils.utils import SemSegCELoss
        parse_res = config.training.parse_map_res
        parse_loss_fn = SemSegCELoss(0.7, parse_res*parse_res//16, ignore_label=0)

    generator = torch.Generator()
    generator.manual_seed(config.training.seed)

    train_loader = DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=True, 
        generator=generator,
    )

    # train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False)

    model.vae.requires_grad_(False)
    model.text_encoder.requires_grad_(False)
    model.unet.requires_grad_(False)

    target_img = target_data["image"]
    target_img = image2tensor(target_img, device)
    target_embedding = victim_models(target_img)

    controller = StructureLossAttentionStore(res=res, batch_size=batch_size)

    delta = torch.zeros(1, 4, res//8, res//8).to(device)
    # delta = torch.randn(1, 4, res//8, res//8).to(device) * 0.1
    if config.training.optim_algo == 'adamw':
        optimizer = AdamW(delta, lr=lr, weight_decay=weight_decay)
    elif config.training.optim_algo == 'adam':
        optimizer = AdamW(delta, lr=lr, weight_decay=0.0)
    else:
        optimizer = PGD(delta, lr=lr, radius=pgd_radius, use_adaptive_lr=use_adaptive_lr)

    for e in range(n_epoch):
        register_attention_control(model, controller)
        for it, batch in enumerate(train_loader):
            images: torch.Tensor
            batch_diff_latents: torch.Tensor
            images, batch_diff_latents, batch_prompt = batch
            images = images.to(device)
            batch_diff_latents = batch_diff_latents.to(device)
            batch_diff_latents.requires_grad_(False)

            prompt = batch_prompt * 2 # since we must calculate the clean and adv latents together
            prompt_embeds, uncond_embeds = embed_prompt(model, prompt)
            context = torch.cat([uncond_embeds, prompt_embeds])

            # Calculate parse map of clean images
            if lam > 0.0:
                images_parse = F.interpolate(images, size=(parse_res, parse_res), mode='bilinear', align_corners=False) #[-1.0, 1.0]
                parse_maps_labels = parse_model((images_parse + 1) / 2)[0].detach().argmax(1)
                parse_maps_labels[parse_maps_labels == 17] = 0  # denote hair as background, which we ignore in the loss

            input_latents = torch.cat([batch_diff_latents, batch_diff_latents + delta]).detach()
            input_latents.requires_grad_(False)

            # 1. initial dinoising loop to pre-calculate the latents of each timestep
            intermediate_latents = [input_latents]
            with torch.no_grad():
                latents = input_latents
                for i in range(start_step, num_inference_steps):
                    t = model.scheduler.timesteps[i]
                    noise_pred = get_noise_pred(model, latents, t, context, do_classifier_free_guidance, guidance_scale)
                    latents = diffusion_step(latents, noise_pred, model.scheduler, t, num_inference_steps)
                    intermediate_latents.append(latents)
                controller.loss = 0.0
                controller.reset()

            # 2. calculate losses on adv_latents and adv_images
            # calculate adv_loss loss from adv_latent z0
            adv_vae_latents = intermediate_latents[-1][batch_size:] # discard the clean latents
            adv_vae_latents.requires_grad_(True)
            adv_images = latent2image(model, adv_vae_latents, return_tensor=True) # [-1, 1]
            adv_embeddings = victim_models(adv_images)
            adv_loss = alpha * cos_dist_loss(adv_embeddings, target_embedding)

            total_loss = 0.0 + adv_loss

            # calculate pairwise loss
            pairwise_loss = torch.tensor(0.0).to(device)
            if eta > 0.0:
                num_pairs = batch_size * (batch_size - 1) / 2
                for i in range(batch_size):
                    for j in range(i + 1, batch_size):
                        adv_embeddings_i = {k: v[i:i+1, :] for k, v in adv_embeddings.items()}
                        adv_embeddings_j = {k: v[j:j+1, :] for k, v in adv_embeddings.items()}
                        pairwise_loss += eta * cos_dist_loss(adv_embeddings_i, adv_embeddings_j) / num_pairs
                if pairwise_loss_sign == 0:
                    pairwise_loss = -pairwise_loss
                total_loss += pairwise_loss

            # calculate neighborhood structural similarity loss
            neighbor_loss = torch.tensor(0.0).to(device)
            if zeta > 0.0:
                neighbor_loss += zeta * neighborhood_struct_loss(adv_embeddings, target_embedding)
                total_loss += neighbor_loss

            # calculate parse_loss
            parse_loss = torch.tensor(0.0).to(device)
            if lam > 0.0:
                adv_image_parse = F.interpolate(adv_images, size=(parse_res, parse_res), mode='bilinear', align_corners=False) #[0.0, 1.0]
                parse_map_adv = parse_model(adv_image_parse)[0]
                parse_loss = parse_loss_fn(parse_map_adv, parse_maps_labels)
                total_loss += lam * parse_loss
            
            # 3. initialize the gradient as dL/dz0
            grad = torch.autograd.grad(total_loss, adv_vae_latents)[0].detach()

            total_loss = total_loss.detach()
            sa_loss = 0.0 # self-attention loss
            ca_loss = 0.0 # cross-attention loss
            
            # 4. memory efficient gradient calculation
            for i in reversed(list(range(start_step, num_inference_steps))):
                t = model.scheduler.timesteps[i]

                # retreive the pre-calculated latents and set the adv half to requires_grad = True
                latent_idx = i - start_step
                latents_pre_calc = intermediate_latents[latent_idx]
                clean_latents_pre_calc, adv_latents_pre_calc = torch.chunk(latents_pre_calc, 2)
                adv_latents_pre_calc.requires_grad_(True)
                latents_pre_calc = torch.cat([clean_latents_pre_calc, adv_latents_pre_calc])

                noise_pred_pre_calc = get_noise_pred(model, latents_pre_calc, t, context, do_classifier_free_guidance, guidance_scale)
                next_latents = diffusion_step(latents_pre_calc, noise_pred_pre_calc, model.scheduler, t, num_inference_steps)

                step_sa_loss = controller.loss
                step_ca_loss = torch.tensor(0.0)

                if beta > 0.0:
                    if not ca_loss_is_targeted:
                        step_ca_loss = cross_attention_loss(controller, batch_prompt, prompt_type, model.tokenizer, res)
                    else:
                        target_prompt = target_data["prompt"]
                        target_att_map = target_data["att_maps"][i]
                        step_ca_loss = targeted_cross_attention_loss(
                            controller = controller, 
                            prompts = batch_prompt, 
                            prompt_type = prompt_type, 
                            target_prompt = target_prompt, 
                            target_att_map = target_att_map, 
                            tokenizer = model.tokenizer, 
                            res = res,
                            w_ends = ca_loss_w_ends,
                            w_blur = ca_loss_w_blur,
                            kernel_size = ca_loss_kernel_size,
                            kl_3d = kl_3d,
                        )
                
                controller.loss = 0.0
                controller.reset()

                step_loss = beta * step_ca_loss + gamma * step_sa_loss

                step_loss_for_grad = torch.sum(next_latents[batch_size:] * grad) + step_loss # next_grad = d/dx(f_r) * grad + d/dx(step_loss)
                grad = torch.autograd.grad(step_loss_for_grad, adv_latents_pre_calc)[0].detach()

                total_loss += step_loss
                sa_loss += step_sa_loss
                ca_loss += step_ca_loss

                controller.loss = 0.0
                controller.reset()
            
            grad = grad.sum(dim=0, keepdim=True)
            optimizer(grad)
            logger.log(f"epoch{e}, iteration{it}", out=False)
            logger.log((
                f"    adv_loss: {adv_loss.item():.3f}, "
                f"parse_loss: {parse_loss.item()*lam:.3f}, "
                f"pairwise_loss: {pairwise_loss.item():.3f}, "
                f"neighbor_loss: {neighbor_loss.item():.3f}, "
                f"cross_attention_loss: {ca_loss.item()*beta:.3f}, "
                f"self_attention_loss: {sa_loss.item()*gamma:.3f}, "
                f"total_loss: {total_loss.item():.3f}"
            ), out=False)
            logger.log(f"    delta norm, l2: {delta.norm()}, linf: {delta.abs().max()}", out=False)
        
        reset_attention_control(model)

        ckpt_file = os.path.join(log_ckpt_dir, f"delta_epoch{e}.pth")
        torch.save(delta, ckpt_file)
        asr, ssim, lpips = eval(
            model = model, 
            testset = trainset, 
            delta = delta, 
            target_img = test_img, 
            test_models = test_models, 
            save_sample_img = False, 
            ckpt_dir = log_ckpt_dir, 
            calculate_sim = True, 
            config = config, 
            logger = logger
        )
        logger.log(f"epoch{e} trainset asr on id {trainset.identity}: {asr}", out=False)
        logger.log(f"trainset perceptual sim: ssim={ssim:.6f}, lpips={lpips:.6f}", out=False)
        logger.log('', out=False)

    return delta


@torch.enable_grad()
def identity_attack_regular(
    model: DiffusionPipeline,
    trainset: Union[IDLatentPromptDataset, GroupLatentPromptDataset],
    target_data: dict,
    test_img: torch.Tensor,
    victim_models: ModelEnsemble,
    test_models: ModelEnsemble,
    parse_model: BiSeNet,
    log_ckpt_dir: str,
    config: Config,
    logger: DatetimeLogger,
) -> torch.Tensor:
    device = model.device
    num_inference_steps = config.diffusion.diffusion_steps
    start_step = config.diffusion.start_step
    guidance_scale = config.diffusion.guidance_scale
    do_classifier_free_guidance = (guidance_scale > 0.0)
    # clip_min = -1e6
    # clip_max = 1e6

    res = config.dataset.res
    prompt_type = config.dataset.prompt_type

    batch_size = config.training.batch_size
    lr = config.training.lr
    use_adaptive_lr = config.training.use_adaptive_lr
    pgd_radius = config.training.pgd_radius
    weight_decay = config.training.weight_decay
    alpha = config.training.attack_loss_weight
    beta = config.training.cross_attn_loss_weight
    gamma = config.training.self_attn_loss_weight
    n_epoch = config.training.total_update_steps // (len(trainset) // batch_size)
    lam = config.training.semseg_lam
    eta = config.training.pairwise_loss_weight
    zeta = config.training.neighborhood_loss_weight
    pairwise_loss_sign = config.training.pairwise_loss_sign
    ca_loss_is_targeted = config.training.ca_loss_is_targeted
    ca_loss_w_blur = config.training.ca_loss_w_blur
    ca_loss_kernel_size = config.training.ca_gaussian_kernel_size
    ca_loss_w_ends = config.training.ca_loss_is_w_ends
    kl_3d = config.training.kl_3d

    if lam > 0.0:
        from faceutils.utils import SemSegCELoss
        parse_res = config.training.parse_map_res
        parse_loss_fn = SemSegCELoss(0.7, parse_res*parse_res//16, ignore_label=0)

    generator = torch.Generator()
    generator.manual_seed(config.training.seed)

    train_loader = DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=True, 
        generator=generator,
    )

    # train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False)

    model.vae.requires_grad_(False)
    model.text_encoder.requires_grad_(False)
    model.unet.requires_grad_(False)

    target_img = target_data["image"]
    target_img = image2tensor(target_img, device)
    target_embedding = victim_models(target_img)

    controller = StructureLossAttentionStore(res=res, batch_size=batch_size)

    delta = torch.zeros(1, 4, res//8, res//8).to(device)
    delta.requires_grad_(True)

    # delta = torch.randn(1, 4, res//8, res//8).to(device) * 0.1
    if config.training.optim_algo == 'adamw':
        optimizer = AdamW(delta, lr=lr, weight_decay=weight_decay)
    elif config.training.optim_algo == 'adam':
        optimizer = AdamW(delta, lr=lr, weight_decay=0.0)
    else:
        optimizer = PGD(delta, lr=lr, radius=pgd_radius, use_adaptive_lr=use_adaptive_lr)

    for e in range(n_epoch):
        register_attention_control(model, controller)
        for it, batch in enumerate(train_loader):
            images: torch.Tensor
            batch_diff_latents: torch.Tensor
            images, batch_diff_latents, batch_prompt = batch
            images = images.to(device)
            batch_diff_latents = batch_diff_latents.to(device)
            batch_diff_latents.requires_grad_(False)

            prompt = batch_prompt * 2 # since we must calculate the clean and adv latents together
            prompt_embeds, uncond_embeds = embed_prompt(model, prompt)
            context = torch.cat([uncond_embeds, prompt_embeds])

            # Calculate parse map of clean images
            if lam > 0.0:
                images_parse = F.interpolate(images, size=(parse_res, parse_res), mode='bilinear', align_corners=False) #[-1.0, 1.0]
                parse_maps_labels = parse_model((images_parse + 1) / 2)[0].detach().argmax(1)
                parse_maps_labels[parse_maps_labels == 17] = 0  # denote hair as background, which we ignore in the loss

            adv_diff_latents = batch_diff_latents + delta
            input_latents = torch.cat([batch_diff_latents, adv_diff_latents])

            latents = input_latents
            total_loss = torch.tensor(0.0, device=device)
            for i in range(start_step, num_inference_steps):
                t = model.scheduler.timesteps[i]
                noise_pred = get_noise_pred(model, latents, t, context, do_classifier_free_guidance, guidance_scale)
                latents = diffusion_step(latents, noise_pred, model.scheduler, t, num_inference_steps)

                step_sa_loss = controller.loss
                step_ca_loss = torch.tensor(0.0)

                if beta > 0.0:
                    if not ca_loss_is_targeted:
                        step_ca_loss = cross_attention_loss(controller, batch_prompt, prompt_type, model.tokenizer, res)
                    else:
                        target_prompt = target_data["prompt"]
                        target_att_map = target_data["att_maps"][i]
                        step_ca_loss = targeted_cross_attention_loss(
                            controller = controller, 
                            prompts = batch_prompt, 
                            prompt_type = prompt_type, 
                            target_prompt = target_prompt, 
                            target_att_map = target_att_map, 
                            tokenizer = model.tokenizer, 
                            res = res,
                            w_ends = ca_loss_w_ends,
                            w_blur = ca_loss_w_blur,
                            kernel_size = ca_loss_kernel_size,
                            kl_3d = kl_3d,
                        )
                
                controller.loss = 0.0
                controller.reset()

                total_loss += beta * step_ca_loss + gamma * step_sa_loss

            adv_vae_latents = latents[batch_size:]  # discard the clean latents
            adv_images = latent2image(model, adv_vae_latents, return_tensor=True) # [-1, 1]
            adv_embeddings = victim_models(adv_images)
            adv_loss = alpha * cos_dist_loss(adv_embeddings, target_embedding)

            parse_loss = torch.tensor(0.0).to(device)
            if lam > 0.0:
                adv_image_parse = F.interpolate(adv_images, size=(parse_res, parse_res), mode='bilinear', align_corners=False) #[0.0, 1.0]
                parse_map_adv = parse_model(adv_image_parse)[0]
                parse_loss = parse_loss_fn(parse_map_adv, parse_maps_labels)
                total_loss += lam * parse_loss

            total_loss += adv_loss

            # calculate pairwise loss
            pairwise_loss = torch.tensor(0.0).to(device)
            if eta > 0.0:
                num_pairs = batch_size * (batch_size - 1) / 2
                for i in range(batch_size):
                    for j in range(i + 1, batch_size):
                        adv_embeddings_i = {k: v[i:i+1, :] for k, v in adv_embeddings.items()}
                        adv_embeddings_j = {k: v[j:j+1, :] for k, v in adv_embeddings.items()}
                        pairwise_loss += eta * cos_dist_loss(adv_embeddings_i, adv_embeddings_j) / num_pairs
                if pairwise_loss_sign == 0:
                    pairwise_loss = -pairwise_loss
                total_loss += pairwise_loss

            # calculate neighborhood structural similarity loss
            neighbor_loss = torch.tensor(0.0).to(device)
            if zeta > 0.0:
                neighbor_loss += zeta * neighborhood_struct_loss(adv_embeddings, target_embedding)
                total_loss += neighbor_loss
            
            total_loss.backward()
            grad = delta.grad
            optimizer(grad)

            logger.log(f"epoch{e}, iteration{it}", out=False)
            logger.log((
                f"    adv_loss: {adv_loss.item():.3f}, "
                f"total_loss: {total_loss.item():.3f}"
            ), out=False)
            logger.log(f"    delta norm, l2: {delta.norm()}, linf: {delta.abs().max()}", out=False)
        
        reset_attention_control(model)

        ckpt_file = os.path.join(log_ckpt_dir, f"delta_epoch{e}.pth")
        torch.save(delta, ckpt_file)
        asr, ssim, lpips = eval(
            model = model, 
            testset = trainset, 
            delta = delta, 
            target_img = test_img, 
            test_models = test_models, 
            save_sample_img = False, 
            ckpt_dir = log_ckpt_dir, 
            calculate_sim = True, 
            config = config, 
            logger = logger
        )
        logger.log(f"epoch{e} trainset asr on id {trainset.identity}: {asr}", out=False)
        logger.log(f"trainset perceptual sim: ssim={ssim:.6f}, lpips={lpips:.6f}", out=False)
        logger.log('', out=False)

    return delta


@torch.no_grad()
def eval(
    model: DiffusionPipeline, 
    testset: Union[IDLatentPromptDataset, GroupLatentPromptDataset], 
    delta: torch.Tensor, 
    target_img: Image.Image,
    test_models: ModelEnsemble,
    save_sample_img: bool,
    ckpt_dir: str,
    calculate_sim: bool,
    config, 
    logger: DatetimeLogger,
) -> Union[dict, Tuple[dict, float, float]]:
    """
    Tests the supplied delta on the testset and test_models. Returns a dict in the form of {test_model_name: asr_on_model}
    """
    device = model.device
    res = config.dataset.res
    num_inference_steps = config.diffusion.diffusion_steps
    start_step = config.diffusion.start_step
    guidance_scale = config.diffusion.guidance_scale
    do_classifier_free_guidance = (guidance_scale > 0.0)

    # generate test adv images
    eval_batch_size = 1
    testloader = DataLoader(testset, batch_size=eval_batch_size, shuffle=False)
    adv_images = []
    clean_images = []
    for batch in testloader:
        batch_diff_latents: torch.Tensor
        batch_images, batch_diff_latents, batch_prompt = batch
        batch_images = batch_images.to(device)
        clean_images.append(batch_images)
        batch_diff_latents = batch_diff_latents.to(device)
        batch_adv_latents = batch_diff_latents + delta
        batch_adv_images = sample(model, batch_prompt, batch_adv_latents, guidance_scale, num_inference_steps, start_step, return_tensor=True, res=res)
        adv_images.append(batch_adv_images)
    
    adv_images = torch.cat(adv_images, dim=0) # [-1.0, 1.0]
    clean_images = torch.cat(clean_images, dim=0) # [-1.0, 1.0]
    target_img = image2tensor(target_img, device)
    target_embedding = test_models(target_img)

    if calculate_sim:
        ssim = calculate_ssim((clean_images + 1) / 2, (adv_images + 1) / 2)
        ssim = torch.mean(ssim).item()
        lpips = calculate_lpips(clean_images, adv_images)
        lpips = torch.mean(lpips).item()

    asc = {test_model_name:0 for test_model_name in test_models.models.keys()} # attack success count
    success_idx = []

    # test attack result
    for i in range(len(testset)):
        adv_image = adv_images[i][None]
        adv_embedding = test_models(adv_image)

        for test_model_name in test_models.models.keys():
            thres = THRES_DICT[test_model_name][1] # we use the 99 percent confidence threshold
            simi = F.cosine_similarity(target_embedding[test_model_name], adv_embedding[test_model_name])
            if simi > thres:
                asc[test_model_name] += 1
                success_idx.append(i)
    
    # store sample img
    if save_sample_img:
        if not success_idx:
            logger.log("No success images.")
            idx = 0
        else:
            idx = success_idx[0]
        sample_filename = testset.image_filenames[idx]
        if '_' in testset.identity:
            testset_ids = testset.identity.split('_')
            img_id = testset_ids[idx // testset.sub_dataset_len]
        else:
            img_id = testset.identity
        source_image: Image.Image = Image.open(os.path.join(config.dataset.images_root, img_id, sample_filename))
        sample_image = adv_images[idx][None]
        sample_image = tensor2image(sample_image)[0]
        sample_file_path = os.path.join(ckpt_dir, f"adv{img_id}_{sample_filename}")
        sample_image.save(sample_file_path)
        source_image = source_image.resize(sample_image.size)
        source_image.save(os.path.join(ckpt_dir, f"src{img_id}_{sample_filename}"))
        logger.log(f"Saved sample image adv{img_id}_{sample_filename}")
    
    asr = {k: v/len(testset) for k, v in asc.items()}

    if calculate_sim:
        return asr, ssim, lpips
    return asr
    

@torch.enable_grad()
def attack(
    model: DiffusionPipeline,
    trainsets: Union[List[IDLatentPromptDataset], List[GroupLatentPromptDataset]],
    testsets: Union[List[IDLatentPromptDataset], List[GroupLatentPromptDataset]],
    target_data: dict,
    victim_models: ModelEnsemble = None,
    test_models: ModelEnsemble = None,
    parse_model=None, # semantic segmentation model
    config=None,
    logger: DatetimeLogger=None,
):
    # device = model.device
    res = config.dataset.res
    # num_inference_steps = config.diffusion.diffusion_steps
    # start_step = config.diffusion.start_step
    # guidance_scale = config.diffusion.guidance_scale
    # do_classifier_free_guidance = (guidance_scale > 0.0)

    train_timer = Timer()

    ids = []
    id_asrs = []
    id_ssims = []
    id_lpipss = []
    target_img = target_data["image"]
    test_img = Image.open(config.dataset.test_path).resize((res, res))
    test_img = image2tensor(test_img)
    for i in range(len(trainsets)):
        train_timer.tic()

        trainset = trainsets[i]
        testset = testsets[i]

        log_ckpt_dir = os.path.join(logger.log_dir, f"{trainset.identity}")
        os.makedirs(log_ckpt_dir, exist_ok=True)

        logger.log(f"Victim id no {i}: {trainset.identity}")

        if config.training.memory_efficient:
            delta = identity_attack_memory_efficient( # enter training loop
                model=model, 
                trainset=trainset, 
                target_data=target_data, 
                test_img=test_img,
                victim_models=victim_models, 
                test_models=test_models, 
                parse_model=parse_model, 
                log_ckpt_dir=log_ckpt_dir, 
                config=config, 
                logger=logger
            )
        else:
            delta = identity_attack_regular( # enter training loop
                model=model, 
                trainset=trainset, 
                target_data=target_data, 
                test_img=test_img,
                victim_models=victim_models, 
                test_models=test_models, 
                parse_model=parse_model, 
                log_ckpt_dir=log_ckpt_dir, 
                config=config, 
                logger=logger
            )

        asr, ssim, lpips = eval(
            model = model, 
            testset = testset, 
            delta = delta, 
            target_img = test_img, 
            test_models = test_models, 
            save_sample_img = True, 
            ckpt_dir = log_ckpt_dir, 
            calculate_sim = True, 
            config = config, 
            logger = logger    
        )

        ids.append(trainset.identity)
        id_asrs.append(asr)
        id_ssims.append(ssim)
        id_lpipss.append(lpips)

        iter_time = train_timer.toc(hms=False)
        average_time = train_timer.average(hms=False)
        iters_left = len(trainsets) - i - 1

        logger.log(f"testset asr on id {trainset.identity}: {asr}")
        logger.log(f"testset perceptual sim: ssim={ssim:.6f}, lpips={lpips:.6f}")
        logger.log(f"id training time: {sec2hms(iter_time)}. estimated remaining time: {sec2hms(average_time * iters_left)}")
        logger.log('\n')
    
    for i in range(len(ids)):
        logger.log(f"testset asr on id {ids[i]}: {id_asrs[i]}")
        logger.log(f"testset perceptual sim: ssim={id_ssims[i]:.6f}, lpips={id_lpipss[i]:.6f}")
        logger.log()
    
    test_model_names = list(id_asrs[0].keys())
    mean_asr = {test_model_name: 0 for test_model_name in test_model_names}
    for id_asr in id_asrs:
        for test_model_name in test_model_names:
            mean_asr[test_model_name] += id_asr[test_model_name]
    for test_model_name in test_model_names:
        mean_asr[test_model_name] = mean_asr[test_model_name] / len(ids)

    logger.log(f"mean asr: {mean_asr}")
    logger.log(f"mean perceptual sim: ssim={sum(id_ssims)/len(ids):.6f}, lpips={sum(id_lpipss)/len(ids):.6f}")

                            
if __name__ == "__main__":
    pass