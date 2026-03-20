import torch
from typing import Literal


class AdamW():
    def __init__(
        self, 
        variable: torch.Tensor,
        lr: float = 0.002, 
        weight_decay: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999, 
        eps: float = 1e-8,
        warm_up_steps: int = 0,
    ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.warm_up_steps = warm_up_steps

        self.variable = variable
        self.momentum = torch.zeros_like(variable)
        self.mean_square = torch.zeros_like(variable)

        self.step_cnt = 1
    
    @torch.no_grad()
    def update(self, grad: torch.Tensor):
        if self.warm_up_steps and self.step_cnt < self.warm_up_steps:
            lr = self.lr * self.step_cnt / self.warm_up_steps
        else:
            lr = self.lr
        self.momentum = self.beta1 * self.momentum + (1 - self.beta1) * grad
        self.mean_square = self.beta2 * self.mean_square + (1 - self.beta2) * grad * grad

        self.momentum = self.momentum / (1 - self.beta1 ** self.step_cnt)
        self.mean_square = self.mean_square / (1 - self.beta2 ** self.step_cnt)

        update_step = self.momentum / (torch.sqrt(self.mean_square) + self.eps)

        self.variable.sub_(lr * update_step)
        if self.weight_decay > 0.0:
            self.variable.sub_(lr * self.weight_decay * self.variable)

        self.step_cnt += 1
    
    def reset(self):
        self.momentum = torch.zeros_like(self.variable)
        self.mean_square = torch.zeros_like(self.variable)
        self.step_cnt = 1

    @torch.no_grad()
    def __call__(self, grad: torch.Tensor):
        assert grad.shape == self.variable.shape
        self.update(grad)
        return


class SGD():
    def __init__(
        self,
        variable: torch.Tensor,
        lr: float,
        weight_decay: float = 0.01,
        warm_up_steps: int = 0
    ):
        self.variable = variable
        self.lr = lr 
        self.weight_decay = weight_decay
        self.warm_up_steps = warm_up_steps
        self.step_cnt = 1
    
    @torch.no_grad()
    def update(self, grad: torch.Tensor):
        if self.warm_up_steps and self.step_cnt < self.warm_up_steps:
            lr = self.lr * self.step_cnt / self.warm_up_steps
        else:
            lr = self.lr
        update_step = grad + self.weight_decay * self.variable
        self.variable = self.variable.sub_(lr * update_step)
        self.step_cnt += 1
    
    @torch.no_grad()
    def __call__(self, grad: torch.Tensor):
        assert grad.shape == self.variable.shape
        self.update(grad)
        return


class PGD():
    def __init__(
        self, 
        variable: torch.Tensor,
        lr: float,
        radius: float = 0.02,
        norm_type: Literal["l2", "linf"] = "linf",
        warm_up_iters: int = 0,
    ):
        self.variable = variable
        self.lr = lr
        self.radius = radius
        self.norm_type = norm_type
        self.num_iters = 1

        self.warm_up_steps = warm_up_iters
        self.adapted_lr = lr

    @torch.no_grad()
    def projection(self):
        if self.norm_type == "l2":
            flat = self.variable.view(self.variable.size(0), -1)
            l2_norm = flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
            scale = self.radius / l2_norm
            scale = torch.minimum(scale, torch.ones_like(scale))
            self.variable.mul_(scale.view(-1, 1, 1, 1))
        if self.norm_type == "linf":
            self.variable.clamp_(-self.radius, self.radius)

    @torch.no_grad()
    def update(self, grad: torch.Tensor):
        if self.warm_up_steps and self.num_iters < self.warm_up_steps:
            update_step = self.lr * (self.num_iters / self.warm_up_steps) * grad.sign()
        else:
            update_step = self.lr * grad.sign()
        self.variable.sub_(update_step)
        self.projection()

        self.num_iters += 1

    @torch.no_grad()
    def __call__(self, grad: torch.Tensor):
        assert grad.shape == self.variable.shape
        self.update(grad)


class MIFGSM():
    def __init__(
        self, 
        variable: torch.Tensor, 
        lr: float=0.01, 
        radius: float=0.1, 
        norm_type: str="linf", 
        momentum: float=1.0,
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