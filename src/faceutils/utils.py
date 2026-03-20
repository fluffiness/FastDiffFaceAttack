import torch
from PIL import Image
import numpy as np
from typing import List, Union, Tuple, Dict, Optional, Callable
import torch.nn as nn
from diffusers import DiffusionPipeline
from assets.victim_models import irse, ir152, facenet
import torchvision.transforms as tfms
import torch.nn.functional as F
import os


@torch.no_grad()
def image2tensor(images: Union[Image.Image, np.ndarray, torch.Tensor], device):
    """
    Converts a PIL Image/numpy.ndarray (H x W x C) in the range [0, 255] to a 
    torch.Tensor in the shape (1 x C x H x W) in the range [-1.0, 1.0].
    """
    if isinstance(images, torch.Tensor):
        return images.to(device)
    if isinstance(images, Image.Image):
        images = np.array(images)
    images = torch.from_numpy(images).float() / 127.5 - 1
    images = images.permute(2, 0, 1).unsqueeze(0).to(device)
    return images


@torch.no_grad()
def batch_image2tensor(images: List[Image.Image]):
    img_tensors = [image2tensor(image) for image in images]
    return torch.cat(img_tensors)


@torch.no_grad()
def tensor2image(img_tensor: torch.Tensor):
    """
    Converts a torch.Tensor in shape (B x C x H x W) in the range [-1.0, 1.0] to 
    a list of PIL Images.
    """
    img_tensor = (img_tensor / 2 + 0.5).clamp(0, 1)
    images = img_tensor.cpu().permute(0, 2, 3, 1).float().detach().numpy()
    images = (images * 255).astype(np.uint8)
    return [Image.fromarray(image) for image in images]
    

@torch.no_grad()
def image2latent(model: DiffusionPipeline, image: Union[Image.Image, np.ndarray, torch.Tensor], sample=False):
    """
    Converts a PIL Image/numpy.ndarray (H x W x C) in the range [0, 255] to a 
    torch.Tensor in the shape (C x H x W) in the range [-1.0, 1.0],
    then encodes them with vae.
    If the input type is torch.Tensor, it is assumed that it's already in the [-1.0, 1.0] range.
    """
    if type(image) is not torch.Tensor:
        if type(image) is Image.Image:
            image = np.array(image)
        image = torch.from_numpy(image).float() / 127.5 - 1
        image = image.permute(2, 0, 1)
    image = image.unsqueeze(0).to(model.device)
    latent_dist = model.vae.encode(image).latent_dist
    latents = latent_dist.sample() if sample else latent_dist.mean
    latents = latents * 0.18215
    return latents


def latent2image(model: DiffusionPipeline, latents: torch.Tensor, return_tensor=False) -> Union[np.ndarray, torch.Tensor]:
    """
    Decodes the latents into numpy arrays with the shape (B x H x W x C)
    If return_tensor is True, returns the image as a torch.Tensor in the range [-1.0, 1.0]
    """
    latents = 1 / model.vae.config.scaling_factor * latents # 0.18215 = 0.18215
    images = model.vae.decode(latents, return_dict=False)[0] # value range [-1.0, 1.0]
    if return_tensor:
        return images # value range [-1.0, 1.0]
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).float().detach().numpy()
    images = (images * 255).astype(np.uint8)
    return images


@torch.no_grad()
def embed_prompt(model: DiffusionPipeline, prompt: Union[str, List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Given text prompts, returns text embeddings and null embeddings."""
    if isinstance(prompt, str):
        prompt = [prompt]
        batch_size = 1
    else:
        batch_size = len(prompt)
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length = model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    prompt_embeds = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = prompt_embeds.shape[1]
    uncond_input = model.tokenizer(
        [""] * batch_size, 
        padding="max_length", 
        max_length=max_length,
        truncation=True,
        return_tensors="pt"
    )
    uncond_embeds = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    return prompt_embeds, uncond_embeds


def init_latent(latent, model, height, width, batch_size):
    """
    Does 2 things:
    1. If no initial latent is provided, generates random initial noise.
    2. If only a single latent is provided, expand to fit the batch size.
    """
    if latent is None:
        latent = torch.randn((1, model.unet.config.in_channels, height // 8, width // 8)) * model.scheduler.init_noise_sigma
    assert latent.shape[1] == 4 and latent.shape[2] == height // 8 and latent.shape[3] == width // 8
    # If latent.shape[0] == 1, expand latent.shape[0] to batch_size, else does nothing
    latents = latent.expand(batch_size, 4, height // 8, width // 8).to(model.device)
    return latents


def get_noise_pred(model, latents, t, context, do_classifier_free_guidance=True, guidance_scale=3.5):
    # Expand the latents if we are doing classifier free guidance
    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
    # Predict the noise residual
    noise_pred = model.unet(latent_model_input, t, encoder_hidden_states=context).sample
    # Perform guidance
    if do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    return noise_pred


def diffusion_step(
    latents: torch.Tensor, 
    noise_pred: torch.Tensor, 
    scheduler,
    t: int,
    num_inference_steps: int,
) -> torch.Tensor:
    
    num_train_timesteps = scheduler.config.num_train_timesteps
    prev_t = max(1, t - (num_train_timesteps // num_inference_steps))  # t-1
    alpha_t = scheduler.alphas_cumprod[t.item()]
    alpha_t_prev = scheduler.alphas_cumprod[prev_t]
    predicted_x0 = (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
    direction_pointing_to_xt = (1 - alpha_t_prev).sqrt() * noise_pred

    return alpha_t_prev.sqrt() * predicted_x0 + direction_pointing_to_xt


# Sample function (regular DDIM)
@torch.no_grad()
def sample(
    model: DiffusionPipeline,
    prompt: Union[str, List[str]],
    start_latents: torch.Tensor = None,
    guidance_scale: float = 3.5,
    num_inference_steps: int = 50,
    start_step: int = 0,
    return_latents: bool = False,
    return_tensor: bool = False,
    do_classifier_free_guidance: bool = True,
    res: int = 512,
):
    """
    Return type:
        If return_latents, returns (all_latents, images), otherwise returns images
    The type of images is torch.Tensor if return_tensor with shape (B, C, H, W).
    The value range of the tensor is [-1.0, 1.0].
    Otherwise, the type List[Image.Image]
    """
    device = model.device

    prompt_embeds, uncond_embeds = embed_prompt(model, prompt)
    context = torch.cat([uncond_embeds, prompt_embeds])

    batch_size = prompt_embeds.shape[0]

    model.scheduler.set_timesteps(num_inference_steps, device=device)

    # Create a random starting point if we don't have one already
    start_latents = init_latent(start_latents, model, res, res, batch_size)
    latents = start_latents
    all_latents = [latents]

    # print("Sampling...")
    for i in range(start_step, num_inference_steps):
        t = model.scheduler.timesteps[i]
        noise_pred = get_noise_pred(model, latents, t, context, do_classifier_free_guidance, guidance_scale)
        latents = diffusion_step(latents, noise_pred, model.scheduler, t, num_inference_steps)
        all_latents.append(latents)

    # Post-processing
    images = latent2image(model, latents, return_tensor) # [-1, 1] if return_tensor, else [0, 255]
    if not return_tensor:
        images = [Image.fromarray(image) for image in images]
    if return_latents:
        return images, all_latents 
    else: 
        return images


def load_test_models(model_names, base_path, device='cuda'):
    """
    Loads pretrained models.
    Returns a dict in which the keys are the model names and the value is a list of [input_res, model]
    """
    test_models = {}
    for model_name in model_names:
        if model_name == 'ir152':
            test_models[model_name] = []
            test_models[model_name].append((112, 112))
            fr_model = ir152.IR_152((112, 112))
            fr_model.load_state_dict(torch.load(os.path.join(base_path, 'ir152.pth')))
            fr_model.to(device)
            fr_model.eval()
            fr_model.requires_grad_(False)
            test_models[model_name].append(fr_model)
        if model_name == 'irse50':
            test_models[model_name] = []
            test_models[model_name].append((112, 112))
            fr_model = irse.Backbone(50, 0.6, 'ir_se')
            fr_model.load_state_dict(torch.load(os.path.join(base_path, 'irse50.pth')))
            fr_model.to(device)
            fr_model.eval()
            fr_model.requires_grad_(False)
            test_models[model_name].append(fr_model)
        if model_name == 'facenet':
            test_models[model_name] = []
            test_models[model_name].append((160, 160))
            fr_model = facenet.InceptionResnetV1(num_classes=8631, device=device)
            fr_model.load_state_dict(torch.load(os.path.join(base_path, 'facenet.pth')))
            fr_model.to(device)
            fr_model.eval()
            fr_model.requires_grad_(False)
            test_models[model_name].append(fr_model)
        if model_name == 'mobile_face':
            test_models[model_name] = []
            test_models[model_name].append((112, 112))
            fr_model = irse.MobileFaceNet(512)
            fr_model.load_state_dict(torch.load(os.path.join(base_path, 'mobile_face.pth')))
            fr_model.to(device)
            fr_model.eval()
            fr_model.requires_grad_(False)
            test_models[model_name].append(fr_model)
    return test_models


class ModelEnsemble(nn.Module):
    """
    A collection of FR models.
    Takes input images of shape (B, C, H, W), with values in [-1, 1].
    The output is a dict with items {model_name: feature}.
    The forward function will automatically resize the input images to the appropriate dimension for each FR model.
    """
    
    def __init__(self, models, device):
        super(ModelEnsemble, self).__init__()
        self.models = models
    
    def forward(self, x, model_names: List[str]=[], defense_fn: Callable=None, save_defense=False) -> dict: # returns a dict of {model_name: batch_embedding}
        all_model_names = list(self.models.keys())
        if model_names:
            assert set(model_names).issubset(all_model_names)
        else:
            model_names = all_model_names

        features = {}

        for model_name in model_names:
            input_size = self.models[model_name][0]
            fr_model = self.models[model_name][1]
            x_resize = F.interpolate(x, size=input_size, mode='bilinear')
            if defense_fn:
                x_resize = defense_fn(x_resize)
                if save_defense:
                    x_save_list = tensor2image(x_resize)
                    for i, x_save in enumerate(x_save_list):
                        x_save.save(f"defense_sample{i}.png")
                    exit()
            emb = fr_model(x_resize)
            features[model_name] = emb

        return features


def cos_simi(emb_1, emb_2):
    return torch.mean(F.cosine_similarity(emb_1, emb_2))


def cos_dist_loss(src_features: Dict[str, torch.Tensor], tgt_feature: Dict[str, torch.Tensor]) -> torch.Tensor: 
    """
    Calculates cos_dist ,i.e. (1 - cos_simi), between source and target features 
    produced by (possibly) multiple target models.
    That is, both source_feature and target_feature are outputs of VictimModels.
    Items in both input dicts are of the format {model_name: batch_embedding}.
    """
    cos_dist_list = []
    for model_name in src_features.keys():
        cos_dist = 1 - cos_simi(src_features[model_name], tgt_feature[model_name].detach())
        cos_dist_list.append(cos_dist)
    cos_loss = torch.mean(torch.stack(cos_dist_list))
    return cos_loss


THRES_DICT = {'ir152': (0.094632, 0.166788, 0.227922), 'irse50': (0.144840, 0.241045, 0.312703),
            'facenet': (0.256587, 0.409131, 0.591191), 'mobile_face': (0.183635, 0.301611, 0.380878), 
            'cosface': (0.144840, 0.241045, 0.312703), 'arcface': (0.144840, 0.241045, 0.312703)}


def js_div(P: torch.Tensor, Q: torch.Tensor, eps=1e-6):
    """
    P, Q: [B, L] a batch (B) of distributions (L)
    Returns:
    Scalar JS divergence (averaged over batch)
    """
    M = 0.5 * (P + Q)
    js = 0.5 * (F.kl_div(M.log(), P, reduction='batchmean') +
                F.kl_div(M.log(), Q, reduction='batchmean'))
    return js


def neighborhood_struct_loss(
        src_features: Dict[str, torch.Tensor], 
        tgt_feature: Optional[Dict[str, torch.Tensor]],
    ): 
    """
    Both source_feature and target_feature are outputs of VictimModels.
    promotes consistent output among the victim models.
    """
    model_names = list(src_features.keys())
    cos_sims_ensemble = []
    for mn in model_names:
        batch_src_f = src_features[mn]
        batch_src_f_norm = batch_src_f / torch.norm(batch_src_f, dim=1, keepdim=True) # (B, L)
        tgt_f = tgt_feature[mn]
        tgt_f_norm = tgt_f / torch.norm(tgt_f, dim=1, keepdim=True) # (1, L)
        cos_sims = torch.sum(batch_src_f_norm*tgt_f_norm, dim=1) # (B)
        # normalize by model thresholds
        cos_sims = cos_sims / THRES_DICT[mn][1] # use 99% confidence threshold
        cos_sims_ensemble.append(cos_sims)
    cos_sims_ensemble = torch.stack(cos_sims_ensemble) # (K, B), K=num_train_models
    return torch.mean(torch.var(cos_sims_ensemble, dim=0))


def neighborhood_struct_loss_old(
        src_features: Dict[str, torch.Tensor], 
        tgt_feature: Optional[Dict[str, torch.Tensor]]=None,
        temp: float=0.1
    ): 
    """
    Both source_feature and target_feature are outputs of VictimModels.
    promotes consistent output among the victim models.
    To strong a constraint
    """
    model_names = list(src_features.keys())

    # Calculate similarity matrices of each FR model
    sim_matrices = []
    for mn in model_names:
        batch_src_f = src_features[mn]
        batch_src_f_norm = batch_src_f / torch.norm(batch_src_f, dim=1, keepdim=True)
        sim_matrix = torch.matmul(batch_src_f_norm, batch_src_f_norm.T)
        model_thres = THRES_DICT[mn][1] # we use the 99% confidence threshold
        model_temp = temp * model_thres
        sim_softmax = F.softmax(sim_matrix/model_temp, dim=1)
        sim_matrices.append(sim_softmax)
    
    # enforce output structural similarity among FR models
    cnt = 0
    total_js = 0.0
    num_models = len(model_names)
    for i in range(num_models):
        for j in range(i+1, num_models):
            js = js_div(sim_matrices[i], sim_matrices[j])
            total_js += js
            cnt += 1
    # print(total_js)
    return total_js / cnt
        

class SemSegCELoss(nn.Module):
    """
    thresh: only losses above thresh is considered. 
            That is, if the loss at a pixel is smaller than thresh, it is considered similar enough to the original
    n_min: if less then n_min pixels have losses larger than thresh, the largest n_min losses is counted towards the total.
    
    First CE loss at each pixel is calculated -> (N, H, W)
    Then, the pixels are flattend and sorted -> (N, HxW)
    """
    def __init__(self, thresh, n_min, ignore_label=255, *args, **kwargs):
        # DiffProtect original setting: (thresh=0.7, n_min=256 * 256 // 16, ignore_label=0)
        super(SemSegCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_label = ignore_label
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_label, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels)
        loss = loss.reshape(N, -1) # flatten the pixels
        loss, _ = torch.sort(loss, dim=1, descending=True)
        img_losses = []
        for i in range(N):
            img_loss = loss[i]
            if img_loss[self.n_min] > self.thresh:
                img_loss =img_loss[img_loss>self.thresh]
            else:
                img_loss = img_loss[:self.n_min]
            img_losses.append(img_loss)
        loss = torch.concat(img_losses)
        return torch.mean(loss)


# for colorization of segmentation maps
PART_COLORS = [[0, 0, 0], [255, 85, 0], [255, 170, 0],
                [255, 0, 85], [255, 0, 170],
                [0, 255, 0], [85, 255, 0], [170, 255, 0],
                [0, 255, 85], [0, 255, 170],
                [0, 0, 255], [85, 0, 255], [170, 0, 255],
                [0, 85, 255], [0, 170, 255],
                [255, 255, 0], [255, 255, 85], [255, 255, 170],
                [255, 0, 255], [255, 85, 255], [255, 170, 255],
                [0, 255, 255], [85, 255, 255], [170, 255, 255]]


def vis_parsing_maps(original_img: Image.Image, parsing_anno: torch.Tensor):
    """
    Visualizes segmentation maps.
    original image: Image.Image in (h, w, c)
    parsing_anno: torch.Tensor of dimension (h, w), 
                  each entry is an integer in the range [0, 18],
                  representing the annotaion of the corresponding pixel.
    """

    original_img = np.array(original_img)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_class = np.max(vis_parsing_anno)

    for pi in range(0, num_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = PART_COLORS[pi]
    
    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    combined = np.concatenate((vis_parsing_anno_color, original_img), axis=1)
    parse_map_vis = Image.fromarray(vis_parsing_anno_color)
    combined_img = Image.fromarray(combined)

    return parse_map_vis, combined_img


def latent_diff_boundary(latent: torch.Tensor, sharpness: float=100)->torch.Tensor:
    """
    Limits the range of the input latent roughly to the range of [0, 1]. Acts as a differentiable version of torch.clamp.
    sharpness controles the size of the transition zone and the sharpness of the transition. 
    The actual range is [-1/sharpness, 1 + 1/sharpness].
    """
    lower = torch.tanh(sharpness * latent) / sharpness
    middle = latent
    upper = (torch.tanh(sharpness * (latent - 1)) / sharpness) + 1
    result = torch.where(
        latent < 0,
        lower,
        torch.where(
            latent < 1,
            middle,
            upper
        )
    )
    return result


def get_mean_asr(id_asrs: List[Dict[str, float]]):
    test_model_names = list(id_asrs[0].keys())
    mean_asr = {test_model_name: 0 for test_model_name in test_model_names}
    for id_asr in id_asrs:
        for test_model_name in test_model_names:
            mean_asr[test_model_name] += id_asr[test_model_name]
    for test_model_name in test_model_names:
        mean_asr[test_model_name] = mean_asr[test_model_name] / len(id_asrs)
    
    return mean_asr


def monitor_gpu_memory(logger, position: str=""):
    allocated_memory = torch.cuda.memory_allocated()
    reserved_memory = torch.cuda.memory_reserved()
    peak_memory = torch.cuda.max_memory_allocated()
    logger.log(f"{position}:")
    logger.log(f"Allocated memory: {allocated_memory / (1024 ** 2):.2f} MB")
    logger.log(f"Reserved memory: {reserved_memory / (1024 ** 2):.2f} MB")
    logger.log(f"Peak allocated usage: {peak_memory / (1024 ** 2):.2f} MB")