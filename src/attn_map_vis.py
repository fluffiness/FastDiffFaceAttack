from faceutils.inversions import accelerated_invert
from faceutils.utils import image2tensor, image2latent, latent2image, tensor2image, sample, get_noise_pred, diffusion_step, embed_prompt
from faceutils.datasets import generate_prompt
from faceutils.attention_control import AttentionStore
from faceutils.attention_control_utils import register_attention_control, reset_attention_control, aggregate_attention
from faceutils.constants import AGE_GENDER_RACE_MAP

from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch
import numpy as np

from PIL import Image, ImageDraw, ImageFont
import os
import json

img_paths = [
    "../datasets/celebamaskhq_identity_dataset/imgs/80/751.jpg",
    "../datasets/celebamaskhq_identity_dataset/imgs/2131/15544.jpg",
    "../datasets/celebamaskhq_identity_dataset/imgs/26/2426.jpg",
    "../datasets/celebamaskhq_identity_dataset/imgs/47/2025.jpg",
    "../datasets/celebamaskhq_identity_dataset/imgs/80/318.jpg",
    "../datasets/face_imgs_in_thesis/452/577.jpg",
    "../datasets/face_imgs_in_thesis/452/5271.jpg",
    "../datasets/face_imgs_in_thesis/452/7124.jpg",
    "../datasets/face_imgs_in_thesis/452/7582.jpg",
    "../datasets/face_imgs_in_thesis/target/085807.jpg"
]
img_path = img_paths[9]

id_attribute_path = "../datasets/celebamaskhq_identity_dataset/annotations/id_attributes.json"
attribute_path = "../datasets/celebamaskhq_identity_dataset/annotations/attributes.json"
img_id = img_path.split('/')[-2]
img_file = img_path.split('/')[-1]
img_no = os.path.splitext(img_file)[0]
res = 512

original_prompt = "A rabbit and a carrot."
edit_prompt = "A dog and a cat"

prompts = [original_prompt, edit_prompt]
start_step = 0
num_inference_steps = 15
guidance_scale = 12

device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")
pretrained_diffusion_path = "stabilityai/stable-diffusion-2-base"

pipe = StableDiffusionPipeline.from_pretrained(pretrained_diffusion_path).to(device)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.scheduler.set_timesteps(num_inference_steps, device)

controller = AttentionStore(res=res)

diff_latent = torch.randn((1, 4, 64, 64)).to(device)

register_attention_control(pipe, controller)

print("sampling...")
prompt_embeds, uncond_embeds = embed_prompt(pipe, prompts)
context = torch.cat([uncond_embeds, prompt_embeds])
intermediate_latents = []
att_records = {}
aggregated_maps = {}
edit_aggregated_maps = {}
# att_map_imgs = {}
with torch.no_grad():
    latents = torch.cat([diff_latent] * 2, dim=0)
    for i in range(start_step, num_inference_steps):
        t = pipe.scheduler.timesteps[i]
        noise_pred = get_noise_pred(pipe, latents, t, context, True, guidance_scale)
        latents = diffusion_step(latents, noise_pred, pipe.scheduler, t, num_inference_steps)
        intermediate_latents.append(latents)
        # print(f"controller.cur_step at t={t}: {controller.cur_step}")
        att_records[int(t)] = controller.get_average_attention()
        aggregated_maps[int(t)] = aggregate_attention(1, controller, res//32, ("up", "down"), True, 0, False)
        edit_aggregated_maps[int(t)] = aggregate_attention(1, controller, res//32, ("up", "down"), True, 1, False)
        controller.loss = 0.0
        controller.reset()

resampled_latents = intermediate_latents[-1]
# print("resampled_latents.shape: ", resampled_latents.shape)

resampled_imgs = latent2image(pipe, resampled_latents)
resampled_img = Image.fromarray(resampled_imgs[0])
edit_img = Image.fromarray(resampled_imgs[1])

# for t, att_store in att_records.items():
#     print(f"At timestep {t}")
#     print("aggregated_maps[t].shape: ", aggregated_maps[t].shape)
    # for k, v in att_store.items():
    #     for att_map in v:
    #         print(f"    {k}: {att_map.shape}")

reset_attention_control(pipe)

# #################################################
# # testing retrieve_word_map function
# tmp_t = int(pipe.scheduler.timesteps[start_step])

# for k, v in att_records[tmp_t].items():
#     print(k, ": ")
#     for m in v:
#         print(f"    {m.shape}")

# att_map_tmp = aggregated_maps[tmp_t]
# print("att_map_tmp.shape: ", att_map_tmp.shape)
# from faceutils.attention_loss import retrieve_word_map
# face_map = retrieve_word_map(att_map_tmp, [original_prompt], "face", pipe.tokenizer)
# print("face_map.shape: ", face_map.shape)
# face_map = 255 * face_map.squeeze() / face_map.max()
# face_map = face_map.detach().cpu().numpy().astype(np.uint8)
# face_map = Image.fromarray(face_map).resize((256, 256))
# face_map.save(f"face_map_at_{tmp_t}.png")
# #################################################


im_save_dir = f"../logs/attn_map_vis/from_scratch_{original_prompt.replace(' ', '_')}"
os.makedirs(im_save_dir, exist_ok=True)
resampled_img.save(os.path.join(im_save_dir, f"sampled_image_ss={start_step}.png"))
edit_img.save(os.path.join(im_save_dir, f"edit_face_ss={start_step}.png"))


# Generate att map images
tokens = pipe.tokenizer.encode(original_prompt)
decoder = pipe.tokenizer.decode

att_im_save_dir = os.path.join(im_save_dir, "attn_maps")
os.makedirs(att_im_save_dir, exist_ok=True)
for t, aggregated_map in aggregated_maps.items():
    t_dir = os.path.join(att_im_save_dir, f"{t:03d}")
    os.makedirs(t_dir, exist_ok=True)
    for i in range(len(tokens)):
        token = tokens[i]
        im_tensor = aggregated_map.squeeze()[:, :, i]
        im_tensor = 255 * im_tensor / im_tensor.max()
        att_map_im = im_tensor.detach().cpu().numpy().astype(np.uint8)
        att_map_im = Image.fromarray(att_map_im).resize((256, 256))
        w, h = att_map_im.width, att_map_im.height
        band_h = 40
        captioned_im = Image.new('L', (w, h + band_h), color=255)
        captioned_im.paste(att_map_im, (0, 0))
        draw = ImageDraw.Draw(captioned_im)
        token_text = decoder(int(token))
        att_map_im.save(os.path.join(t_dir, f"token{i}_{token_text}_uncap.png"))
        draw.text((10, h+10), token_text, fill=0, font=ImageFont.load_default())
        captioned_im.save(os.path.join(t_dir, f"token{i}_{token_text}.png"))


register_attention_control(pipe, controller)
print("sampling...again")
with torch.no_grad():
    latents = torch.cat([diff_latent] * 2, dim=0)
    for i in range(start_step, num_inference_steps):
        t = pipe.scheduler.timesteps[i]
        noise_pred = get_noise_pred(pipe, latents, t, context, True, guidance_scale)
        latents = diffusion_step(latents, noise_pred, pipe.scheduler, t, num_inference_steps)
        intermediate_latents.append(latents)

averaged_att_map = aggregate_attention(1, controller, res//32, ("up", "down"), True, 0, False)

averaged_dir = os.path.join(att_im_save_dir, "averaged")
os.makedirs(averaged_dir, exist_ok=True)
for i in range(len(tokens)):
    token = tokens[i]
    im_tensor = aggregated_map.squeeze()[:, :, i]
    im_tensor = 255 * im_tensor / im_tensor.max()
    att_map_im = im_tensor.detach().cpu().numpy().astype(np.uint8)
    att_map_im = Image.fromarray(att_map_im).resize((256, 256))
    w, h = att_map_im.width, att_map_im.height
    band_h = 40
    captioned_im = Image.new('L', (w, h + band_h), color=255)
    captioned_im.paste(att_map_im, (0, 0))
    draw = ImageDraw.Draw(captioned_im)
    token_text = decoder(int(token))
    att_map_im.save(os.path.join(averaged_dir, f"token{i}_{token_text}_nocap.png"))
    draw.text((10, h+10), token_text, fill=0, font=ImageFont.load_default())
    captioned_im.save(os.path.join(averaged_dir, f"token{i}_{token_text}.png"))

reset_attention_control(pipe)