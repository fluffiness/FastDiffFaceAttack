from typing import Union, Tuple
import torch
import abc

"""
Dimension of attention maps:
    self attention: (Bxh, HxW, HxW)
    cross attention: (Bxh, HxW, L)

    B: batch size
    h: number of heads of the attention module
    H, W: height and width of the latents
        i.e. the latents in this layer in the unet is of the shape (B, C, H, W)
    L: tokenizer.model_max_length (77 by default) (for cross attention) or HxW (for self attention)
"""

class AttentionControl(abc.ABC):
    def __init__(self):
        self.cur_step = 0
        # total number of attention layers in the model
        self.num_att_layers = -1 # asigned during the register_attention_control function
        # number of processed attention layers
        self.cur_att_layer = 0

    def between_steps(self):
        return

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str): # place_in_unet is in ["up", "mid", "down"]
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str): # place_in_unet is in ["up", "mid", "down"]
        if self.cur_att_layer >= 0:
            b = attn.shape[0] # batch size
            # when using cfg, the first half of the batch is for unconditional generation, 
            # while the second half of the batch is for conditional generation
            self.forward(attn[b // 2:], is_cross, place_in_unet)

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers: # all attention layers processed
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn


class AttentionStore(AttentionControl):
    def __init__(self, res):
        # res: resolution of input image
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.res = res

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        # place in unet is in ["up", "mid", "down"]
        # Stores the attention maps in self.step_store
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        # There are 4 types of cross attention maps, with resolutions res//8, res//16, res//32., and res//64.
        # To prevent memory overhead, only those of resolutions res//16, res//32, and res//64 are stored.
        # res//64 only appears in place_in_unet=mid, while the others appear in place_in_unet=up or place_in_unet=down.
        # Note that res//64 matches the dimensions of the diffusion latent code
        if attn.shape[1] <= (self.res // 16) ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        # map_type = 'cross' if is_cross else 'self'
        # print(f"Storing {map_type} attn map of shape {attn.shape}")
        # self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        # Sums the attention maps in self.step_store to the corresponding maps in self.attention_store,
        # then resets self.step_store
        # The maps from each dinoising step is not stored separately, 
        # which does not matter in our case since we calculate the losses one step at a time either way
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] = self.step_store[key][i] + self.attention_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        # Before reset is explicitly called, attention maps from each denoising step is added together.
        # Calling get_average_attention returns the attention maps averaged over the number of steps taken.
        # If we reset at each step, then this just returns the self.step_store.
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        # del self.step_store
        # del self.attention_store
        self.step_store = self.get_empty_store()
        self.attention_store = {}


class StructureLossAttentionStore(AttentionStore, abc.ABC):
    def __init__(self, res=512, batch_size=1):
        """
        Calcualtes self-attention loss in addition to storing attention maps.
        res: resolution of training/testing images
        batch_size: training/testing batch size
        """
        super(StructureLossAttentionStore, self).__init__(res)
        self.batch_size = batch_size
        self.loss = 0
        self.criterion = torch.nn.MSELoss()

    def forward(self, attn, is_cross: bool, place_in_unet: str): # place_in_unet is in ["up", "mid", "down"]
        # During training, we pass latents = torch.cat([original_latents, adversarial_latents])
        # Therefore, the shape of the input attn is (2Bxh, HxW, L)
        # Note that the unconditional half is already discarded in the __call__ method in AttentionControl
        super(StructureLossAttentionStore, self).forward(attn, is_cross, place_in_unet)

        if not is_cross:
            """
                ==========================================
                ========= Self Attention Control =========
                ==========================================
            """
            b = self.batch_size
            h = attn.shape[0] // (2 * b) # number of heads
            attn = attn.reshape(2 * b, h, *attn.shape[1:]) # (2B, h, HxW, L)
            attn_base, attn_adv = attn[:b], attn[b:] # split the original and adversarial latents, each (B, h, HxW, L)

            self.loss += self.criterion(attn_base, attn_adv)
            attn = attn.reshape(2 * b * h, *attn.shape[2:]) # (2Bxh, HxW, L)

        return attn
