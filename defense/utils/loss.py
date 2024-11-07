import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from monai.utils import LossReduction

class ShadowLoss(_Loss):
    def __init__(
        self,
        cfg,
        reduction = LossReduction.MEAN,
    ):
        super().__init__(reduction=LossReduction(reduction).value)
        self.cfg = cfg

    def img_total_var(self, inputs_jit):                                                               
        # COMPUTE total variation regularization loss
        diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
        diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
        diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
        diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]
        loss_var_l2 = (
            torch.norm(diff1)
            + torch.norm(diff2)
            + torch.norm(diff3)
            + torch.norm(diff4)
        )
        return loss_var_l2

    def forward(self, pseudo_img, grad_list, shadow_grad, bn_hooks, inputs):

        loss = 0.
        # gradient matching
        grad_matching_loss = 0.
        for i in range(len(grad_list)):
            grad_matching_loss += torch.norm(shadow_grad[i] - grad_list[i].to(shadow_grad[i].device))
        grad_matching_loss = self.cfg.shadow_grad_match * grad_matching_loss

        # regularization terms
        # img prior
        loss_total_var = self.cfg.shadow_total_var * self.img_total_var(pseudo_img)

        # img norm
        loss_norm = torch.norm(pseudo_img, 2) / ((pseudo_img.shape[2]*pseudo_img.shape[3]) ** 2)
        loss_norm = self.cfg.shadow_img_norm * loss_norm

        # BN
        loss_bn = 0.
        for hook in bn_hooks:
            loss_bn += hook.r_feature
            hook.close()
        loss_bn = self.cfg.shadow_bn * loss_bn

        # rec
        loss_rec = nn.MSELoss()(pseudo_img, inputs)
        loss_rec = self.cfg.shadow_rec * loss_rec

        loss = grad_matching_loss + loss_total_var + loss_bn + loss_norm + loss_rec 

        return loss, grad_matching_loss.item(), loss_bn.item(), loss_total_var.item(), loss_norm.item(), loss_rec.item()
    
class Rec_Loss(_Loss):
    def __init__(
        self,
        cfg,
        reduction = LossReduction.MEAN,
    ):
        super().__init__(reduction=LossReduction(reduction).value)
        self.cfg = cfg
        self.mse_loss = nn.MSELoss()

    def forward(self, pseudo_img, img):
        loss = self.mse_loss(pseudo_img, img).mean()
        return loss

class Rec_Loss_nomean(_Loss):
    def __init__(
        self,
        cfg,
        reduction = LossReduction.MEAN,
    ):
        super().__init__(reduction=LossReduction(reduction).value)
        self.cfg = cfg
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, pseudo_img, img):
        loss = self.mse_loss(pseudo_img, img)
        return loss