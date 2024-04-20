import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
e = IPython.embed

class ACTPolicy(nn.Module):#this is the ACT model, consisting of lots of neural networks, convolutional or transformers
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']#10 or 100 something
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None):#forward? Is it the pytorch2 version of forward in pytorch1? seems not
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        #print("before:", qpos.size(), image.size())
        #torch.Size([1, 14]),torch.Size([1, 1, 3, 480, 640]) in validation in both TE and not TE
        #torch.Size([8, 14]) torch.Size([8, 1, 3, 480, 640]) in traning(8 may also be 2) since 10-8=2
        if actions is not None: # training time
            #if actions.shape[0]==100, then you can comment the following 2 lines!
            #actions = actions[:, :self.model.num_queries]#the first dim is the batch size, the second dim is the effective length of the action
            #is_pad = is_pad[:, :self.model.num_queries]#only the first self.model.num_queries is needed!
            #print("all the shapes put into the network:",qpos.size(),image.size(),actions.size(),is_pad.size())
            #torch.Size([8, 14]),torch.Size([8, 1, 3, 480, 640]),torch.Size([8, 100, 14]),torch.Size([8, 100])#8 may also be 2
            #torch.Size([8, 7]) torch.Size([8, 2, 3, 240, 320]) torch.Size([8, 100, 7]) torch.Size([8, 100])
            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)#this is the forward method of the model!
            #print("train: action,ahat,is_pad,mu,logvar:",actions.shape,a_hat.shape,is_pad_hat.shape, mu.shape, logvar.shape)#a_hat is 100, how to get this as the label?
            #torch.Size([8, 100, 7]) torch.Size([8, 100, 7]) torch.Size([8, 100, 1]) torch.Size([8, 32]) torch.Size([8, 32])
            #torch.Size([8, 100, 14]) torch.Size([8, 100, 14]) torch.Size([8, 100, 1]) torch.Size([8, 32]) torch.Size([8, 32])
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')#showing the imitation learning intrinsics of this method
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict#so during training, I need to return the loss for backprop
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
            #print("val: a_hat:",a_hat.shape)#torch.Size([1, 100, 14])
            #val: a_hat: torch.Size([1, 100, 14])#same when both TE or not TE!
            return a_hat#does the above embody that z=0?

    def configure_optimizers(self):#接口, port, such OOP things
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)#it is also imitation learning, as it tries to mimic the expert demonstrations
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
