import torch
import torch.nn.functional as F
from typing import List, Tuple
import re
from faceutils.attention_control import AttentionStore
from faceutils.attention_control_utils import aggregate_attention
from faceutils.constants import KEY_WORDS


def find_word_in_sentence(word_list: List[str], sentence: str):
    """
    Finds words from the list that are in the sentence. 
    Words are case-sensitive.
    Returns them as a list, sorted by order of first occurence.

    """
    words_in_sentence = re.findall(r'\b\w+\b', sentence)
    seen = {}
    for idx, w in enumerate(words_in_sentence):
        if w in word_list and w not in seen:
            seen[w] = idx  # Store first occurrence index
    
    sorted_words = sorted(seen, key=seen.get)
    return sorted_words


def reduce_att_map(attn_map: torch.Tensor, prompt: str, words: List[str], tokenizer):

    """
    needs att_map to be cross attention map with start and end token already removed
    """

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
        word_map = attn_map[:, :, :, start_idx: end_idx] # (B, H, W, word_emb_len)
        reduced_map_components.append(attn_map[:, :, :, prev_end_idx: start_idx])
        reduced_map_components.append(word_map.sum(dim=-1, keepdim=True))
        prev_end_idx = end_idx
    reduced_map_components.append(attn_map[:, :, :, end_idx:])
    reduced_map = torch.cat(reduced_map_components, dim=-1)

    return reduced_map


def retrieve_word_maps(attn_maps: torch.Tensor, batch_prompt: List[str], words: List[str], tokenizer) -> List[torch.Tensor]:
    """
    attn_maps: a batch of full cross-attention maps, with start and end tokens still attached. Its shape can be (B, H, W, L) or (B, h, HxW, L)
                or (B, x, x, L) in general.
    batch_prompt: a batch of prompts. may be different
    words: a list words of interest. the maps corresponding to these words will be retrieved. can included words not in any prompt
    returns a list of attn_maps in the shape of (1, x, x, L), where the L dimension is the words concatenated together
    since the prompts are difference, L may be different for each map in the list
    """
    batch_word_maps = []
    for prompt_i, prompt in enumerate(batch_prompt):
        sorted_words = find_word_in_sentence(words, prompt)
        prompt_words = prompt.split(' ')
        n_tokens = [0] + [len(tokenizer.encode(word))-2 for word in prompt_words]
        cmul_tokens = [0] * len(n_tokens) # to account for some words consisting of many tokens.
        for token_i in range(1, len(n_tokens)):
            cmul_tokens[token_i] = cmul_tokens[token_i-1] + n_tokens[token_i]
        word_maps = []
        for word in sorted_words:
            word_idx = prompt_words.index(word)
            start_idx = cmul_tokens[word_idx]
            end_idx = cmul_tokens[word_idx+1]
            word_maps.append(attn_maps[prompt_i:prompt_i+1, :, :, start_idx+1: end_idx+1]) # +1 accounts fot the start token
        word_maps = torch.cat(word_maps, dim=-1)
        batch_word_maps.append(word_maps)
    return batch_word_maps


def retrieve_word_maps_single(attn_maps: torch.Tensor, prompt: str, words: List[str], tokenizer) -> torch.Tensor:
    """
    attn_maps: a batch of full cross-attention maps, with start and end tokens still attached. Its shape can be (B, H, W, L) or (B, h, HxW, L)
                or (B, x, x, L) in general.
    prompt: a single prompt. all attn_maps in the batch are generated using the same prompt
    words: a list words of interest. the maps corresponding to these words will be retrieved. can included words not in any prompt
    returns a batch of attn_maps in the shape of (B, H, W, L), where the L dimension is the words concatenated together
    """
    sorted_words = find_word_in_sentence(words, prompt)
    prompt_words = prompt.split(' ')
    n_tokens = [0] + [len(tokenizer.encode(word))-2 for word in prompt_words]
    cmul_tokens = [0] * len(n_tokens) # to account for some words consisting of many tokens.
    for token_i in range(1, len(n_tokens)):
        cmul_tokens[token_i] = cmul_tokens[token_i-1] + n_tokens[token_i]
    word_maps = []
    for word in sorted_words:
        word_idx = prompt_words.index(word)
        start_idx = cmul_tokens[word_idx]
        end_idx = cmul_tokens[word_idx+1]
        word_maps.append(attn_maps[:, :, :, start_idx+1: end_idx+1]) # +1 accounts fot the start token
    word_maps = torch.cat(word_maps, dim=-1)
    return word_maps


def targeted_cross_attention_loss(
    controller: AttentionStore, 
    batch_size: int,
    target_prompt: str,
    target_att_map: torch.Tensor,
    tokenizer,
    config
) -> torch.Tensor:

    res = config.dataset.res

    # before_attention_map = aggregate_attention(batch_size, controller, res//32, ("up", "down"), True, 0, is_cpu=False)
    after_attention_map = aggregate_attention(batch_size, controller, res//32, ("up", "down"), True, 1, is_cpu=False)

    target_emb_len = len(tokenizer.encode(target_prompt)) - 2 # -2 to discard the <|startoftext|> and <|endoftext|> tokens
    target_sot_map = target_att_map[:, :, :, 0:1]
    target_text_map = target_att_map[:, :, :, 1: target_emb_len+1]
    target_eot_map = torch.sum(target_att_map[:, :, :, target_emb_len+1:], dim=-1, keepdim=True)
    target_att_map = torch.cat([target_sot_map, target_text_map, target_eot_map], dim=-1)

    att_maps = after_attention_map
    sot_maps = att_maps[:, :, :, 0:1]
    text_maps = att_maps[:, :, :, 1: target_emb_len+1]
    eot_maps = torch.sum(att_maps[:, :, :, target_emb_len+1:], dim=-1, keepdim=True)
    att_maps = torch.cat([sot_maps, text_maps, eot_maps], dim=-1)

    if config.training.use_mse:
        return torch.mean((att_maps - target_att_map)**2)       

    eps = 1e-8
    if config.training.kl_3d:
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

    P = P.clamp(min=eps)
    Q = Q.clamp(min=eps)

    kl = F.kl_div(P.log(), Q, reduction='none')  # shape: (B, H, W, L)
    kl_per_pixel = kl.sum(dim=-1) # sum over L → shape (B, H, W)
    ca_loss = kl_per_pixel.mean()

    return ca_loss


def cross_attention_loss(
    controller: AttentionStore, 
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


def attn_structural_loss(
    controller: AttentionStore, 
    batch_prompt: List[str],
    tokenizer,
    lam_ca_reg: float = 0.0,
    lam_sa: float = 0.0,
    words: List[str]=None,
    use_mask: bool=True,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Regulates the change of cross attention maps from clean to adv.
    Simulates the effect of semantic segmentation without additional external models.
    Should work better with more complex prompts? -> Not necessarily...
    Also moved self-attention loss here. When using the original structural loss from DiffAttack, 
    self-attn maps of all resolution are incorporated. Since AttentionStore doesn't store all resolutions, here the
    self-attn loss only incorporates im_res//16 and below.
    """

    b = len(batch_prompt) # batch size
    words_of_interest = words if words else KEY_WORDS
    attn_store = controller.get_average_attention()
    # get_average_attention() returns the step_store, which sotres the attn_maps in a dict with format:
    # {place_and_type: [list of maps]}
    # place_and_type ex. up_cross, mid_self, down_cross, etc.

    ca_reg_loss = torch.tensor(0.0).to(device)
    sa_loss = torch.tensor(0.0).to(device)
    for place_and_type, stored_maps in attn_store.items():
        if "cross" in place_and_type and lam_ca_reg > 0.0:
            # print("calculating ca reg loss")
            for attn in stored_maps:
                h = attn.shape[0] // (2 * b) # number of heads
                attn = attn.reshape(2 * b, h, *attn.shape[1:]) # (2B, h, HxW, L)
                attn_base, attn_adv = attn[:b], attn[b:]
                attn_base_words = retrieve_word_maps(attn_base, batch_prompt, words_of_interest, tokenizer)
                attn_adv_words = retrieve_word_maps(attn_adv, batch_prompt, words_of_interest, tokenizer)
                # print("attn_base_words[0].shape: ", attn_base_words[0].shape)
                # exit()
                for i in range(len(attn_base_words)):
                    ca_reg_loss += F.mse_loss(attn_base_words[i], attn_adv_words[i])
        if "self" in place_and_type and lam_sa > 0.0:
            # print("calculating sa loss")
            for i, attn in enumerate(stored_maps):
                h = attn.shape[0] // (2 * b) # number of heads
                if use_mask:
                    corr_base_cross_attn = attn_store[place_and_type.replace("self", "cross")][i]
                    # print("corr_base_cross_attn.shape: ", corr_base_cross_attn.shape)
                    corr_base_cross_attn = corr_base_cross_attn.reshape(2 * b, h, *corr_base_cross_attn.shape[1:])[:b]
                    # print("corr_base_cross_attn.shape: ", corr_base_cross_attn.shape)
                    corr_face_attn = torch.cat(retrieve_word_maps(corr_base_cross_attn, batch_prompt, ["face"], tokenizer), dim=0) # (B, h, HxW, 1)
                    corr_face_attn = torch.sum(corr_face_attn, dim=1, keepdim=True) # (B, 1, HxW, 1)
                    face_attn_mask_1d = corr_face_attn / corr_face_attn.max()
                    face_attn_mask_2d = face_attn_mask_1d * face_attn_mask_1d.transpose(2, 3) # (B, 1, HxW, HxW)
                    # print(f"face_attn_mask_2d.shape: {face_attn_mask_2d.shape}")
                    # exit()
                attn = attn.reshape(2 * b, h, *attn.shape[1:]) # (2B, h, HxW, HxW)
                attn_base, attn_adv = attn[:b], attn[b:] # split the original and adversarial latents, each (B, h, HxW, HxW)
                attn_diff = attn_base - attn_adv
                if use_mask:
                    attn_diff = attn_diff * face_attn_mask_2d
                sa_loss += torch.mean(attn_diff**2)

    return lam_ca_reg*ca_reg_loss, lam_sa*sa_loss