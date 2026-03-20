# requirements
# numpy                     1.21.2
# torch                     1.12.0+cu113
# torchvision               0.13.0+cu113
# timm                      0.4.12
# transformers              4.46.3
# accelerate                1.0.1 

import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
pretrained_diffusion_path = "stabilityai/stable-diffusion-2-base"
res = 512
prompts = [
    "water color painting of a golden retriever on the beach",
    "oil painting of a cat by a window during sunset"
]

pipe = StableDiffusionPipeline.from_pretrained(pretrained_diffusion_path).to(device)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

tokenizer = pipe.tokenizer
tokens = tokenizer.encode(prompts[0]) # takes a str as input
# Takes a str and returns a List[int].
# the first integer represents the <|startoftext|> token
# and the last integer represents the <|endoftext|> token
print(tokens)
word = tokenizer.decode(tokens[1]) # decodes an integer into a str
sentence = tokenizer.decode(tokens) # decodes a List[int] into a str
print(word, sentence)

batch_tokens = tokenizer(prompts) # takes a batch of str as input
# outputs: 
# {
#   "input_ids": List[List[int]],
#   "attention_mask": List[List[int]]
# }
print(batch_tokens.input_ids)


tensor_tokens = tokenizer(
    prompts,
    padding="max_length",
    max_length = tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt",
)
# outputs:
# {
#   "input_ids": torch.Tensor, shape=(B, max_length)
#   "attention_mask": torch.Tensor, shape=(B, max_length)
# }
# past the original sentence, the tokens are padded with zeroes to reach max_length
print(tensor_tokens.input_ids.shape)