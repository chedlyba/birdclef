import numpy as np 
import pandas as pd 
import torch 
import math 
from torch import nn
from collections import OrderedDict
import matplotlib.pyplot as plt 

def binarize(tensor):
    return tensor.sign()

class BinarizeLinearLayer(nn.Linear):
    def __init__(self, is_input=False, *kargs, **kwargs):
        super(BinarizeLinearLayer, self).__init__(*kargs, **kwargs)
        self.is_input = False


    def forward(self, input):
        if not self.is_input:
            input.data = binarize(input.data)
        if not hasattr(self.weight, 'org'): 
            self.weight.org = self.weight.data.clone() 
        self.weight.data = binarize(self.weight.org)
        out = nn.functional.linear(input, self.weight)
        if self.bias is not None:
            self.bias.org = self.bias.data.clone()
            out+= self.bias.view(1, -1).expand_as(out)
        
        return out

class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)
        self.is_input = False

    def forward(self, input):
        
        if not self.is_input:
            input.data = binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=binarize(self.weight.org)

        out = torch.nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out

class SignActivation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i.sign()
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i, = ctx.saved_tensors
        grad_i = grad_output.clone()
        grad_i [ i.abs() > 1.0 ] = 0
        return grad_i

class AudioClassifierBNN(nn.Module):

    def __init__(self, init='uniform', width=0.01, n_labels=10, n_channels=1):
        super(AudioClassifierBNN, self).__init__()
        self.n_channels = n_channels
        self.hidden_layers = 4
        self.n_labels = n_labels
        self.layer_list = [
            ( ('cv1', BinarizeConv2d(self.n_channels, 16, kernel_size=(5, 5), stride = (2, 2), bias=False, padding=(2 ,2))) ),
            ( ('bn1', nn.BatchNorm2d(16)) ),
            ( ('cv2', BinarizeConv2d(16, 64, kernel_size=(3, 3), stride = (2, 2), bias=False, padding=(2 ,2))) ),
            ( ('bn2', nn.BatchNorm2d(64)) ),
            ( ('cv3', BinarizeConv2d(64,32, kernel_size=(3, 3), stride = (2, 2), bias=False, padding=(2 ,2))) ),
            ( ('bn3', nn.BatchNorm2d(32) ) ),
            ( ('ap', nn.AdaptiveAvgPool2d(output_size=1)) ),
            ( ('fc', BinarizeLinearLayer(in_features=32, out_features=n_labels)) ),
            ( ('bn4', nn.BatchNorm1d(n_labels)) )
        ]
        
        self.layers = torch.nn.ModuleDict(OrderedDict(self.layer_list))

        for key in self.layers.keys():
            if not('bn' in key) and not('ap' in key):
                if init == "gauss":
                    torch.nn.init.normal_(self.layers[key].weight, mean=0, std=width)
                if init == "uniform":
                    torch.nn.init.uniform_(self.layers[key].weight, a= -width/2, b=width/2)

    def forward(self, x):
        self.layers['cv1'].is_input = True
        x = self.layers['cv1'](x)
        x = self.layers['bn1'](x)
        x = SignActivation.apply(x)
        x = self.layers['cv2'](x)
        x = self.layers['bn2'](x)
        x = SignActivation.apply(x)
        x = self.layers['cv3'](x)
        x = self.layers['bn3'](x)
        x = self.layers['ap'](x)
        x = x.view(x.shape[0], -1)
        x = SignActivation.apply(x)
        x = self.layers['fc'](x)
        x = self.layers['bn4'](x)
        return x

    def set_n_labels(self, n_labels):
        self.n_labels = n_labels
        self.layer_list[-1] = ('bn4', nn.BatchNorm1d(self.n_labels))
        self.layers = torch.nn.ModuleDict(OrderedDict(self.layer_list))

class Adam_meta(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), meta = 1.35, eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, meta=meta, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam_meta, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam_meta, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
            
                    if len(p.size())!=1:
                        state['followed_weight'] = np.random.randint(p.size(0)),np.random.randint(p.size(1))
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(p.data, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad ,alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_( grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])


                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                

                binary_weight_before_update = torch.sign(p.data)
                condition_consolidation = (torch.mul(binary_weight_before_update, exp_avg) > 0.0 )

                decayed_exp_avg = torch.mul(torch.ones_like(p.data)-torch.pow(torch.tanh(group['meta']*torch.abs(p.data)),2) ,exp_avg)

  
                if len(p.size())==1: # True if p is bias, false if p is weight
                    p.data.addcdiv_(-step_size, exp_avg, denom)
                else:
                    #p.data.addcdiv_(-step_size, exp_avg , denom)  #normal update
                    p.data.addcdiv_(-step_size, torch.where(condition_consolidation, decayed_exp_avg, exp_avg) , denom)  #assymetric lr for metaplasticity
                    
        return loss

class AudioClassifier(nn.Module):

    def __init__(self, init='gauss', width=0.01, n_labels=10, n_channels=1):
        super(AudioClassifier, self).__init__()
        self.n_channels = n_channels
        self.hidden_layers = 4
        self.n_labels = n_labels
        self.layer_list= [
            ( ('cv1', nn.Conv2d(self.n_channels, 16, kernel_size=(5, 5), stride = (2, 2), bias=False, padding=(2 ,2)))) ,
            ( ('bn1', nn.BatchNorm2d(16)) ),
            ( ('cv2', nn.Conv2d(16, 64, kernel_size=(3, 3), stride = (2, 2), bias=False, padding=(2 ,2))) ),
            ( ('bn2', nn.BatchNorm2d(64)) ),
            ( ('cv3', nn.Conv2d(64, 32, kernel_size=(3, 3), stride = (2, 2), bias=False, padding=(2 ,2))) ),
            ( ('bn3', nn.BatchNorm2d(32)) ),
            ( ('ap', nn.AdaptiveAvgPool2d(output_size=1)) ),
            ( ('fc', nn.Linear(in_features=32, out_features=n_labels)) ),
            ( ('bn4', nn.BatchNorm1d(n_labels)) )
        ]
        self.layers = torch.nn.ModuleDict(OrderedDict(self.layer_list))

        for key in self.layers.keys():
            if not('bn' in key) and not('ap' in key):
                if init == "gauss":
                    torch.nn.init.normal_(self.layers[key].weight, mean=0, std=width)
                if init == "uniform":
                    torch.nn.init.uniform_(self.layers[key].weight, a= -width/2, b=width/2)

    def forward(self, x):

        x = self.layers['cv1'](x)
        x = self.layers['bn1'](x)
        x = self.layers['cv2'](x)
        x = self.layers['bn2'](x)
        x = self.layers['cv3'](x)
        x = self.layers['bn3'](x)
        x = self.layers['ap'](x)
        x = x.view(x.shape[0], -1)
        x = self.layers['fc'](x)
        x = self.layers['bn4'](x)
        return x

class Adam_bk(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), n_bk=1, ratios=[0], areas=[1],  meta = 1.35, feedback=0.0, eps=1e-8,
                 weight_decay=0, amsgrad=False, path='.'):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, n_bk=n_bk, ratios=ratios, areas=areas, meta=meta, feedback=feedback, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, path=path)
        super(Adam_bk, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam_bk, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            n_bk = group['n_bk']
            ratios = group['ratios']
            areas = group['areas']
            meta = group['meta']
            feedback = group['feedback']
            path = group['path']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # Initializing beakers
                    for bk_idx in range(n_bk+1):
                        if bk_idx==n_bk:  # create an additional beaker clamped at 0
                            state['bk'+str(bk_idx)+'_t-1'] = torch.zeros_like(p)
                            state['bk'+str(bk_idx)+'_t']   = torch.zeros_like(p)
                        else:             # create other beakers at equilibrium
                            state['bk'+str(bk_idx)+'_t-1'] = torch.empty_like(p).copy_(p)
                            state['bk'+str(bk_idx)+'_t']   = torch.empty_like(p).copy_(p)

                        state['bk'+str(bk_idx)+'_lvl'] = []

                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
            
    
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)  #p.data

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])


                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                if p.dim()==1: # True if p is bias, false if p is weight
                    p.data.addcdiv_(-step_size, exp_avg, denom)
                else:
                    # weight update
                    p.data.addcdiv_(-step_size, exp_avg, denom)
                    p.data.add_((ratios[0]/areas[0])*(state['bk1_t-1']-state['bk0_t-1']))
                    p.data.add_(torch.where( (state['bk'+str(n_bk-1)+'_t-1'] - state['bk0_t-1']) * state['bk'+str(n_bk-1)+'_t-1'].sign() > 0 , feedback*(state['bk'+str(n_bk-1)+'_t-1'] - state['bk0_t-1']),
                                                                                                                      torch.zeros_like(p.data)))
                    # Update of the beaker levels
                    with torch.no_grad():
                        for bk_idx in range(1, n_bk):
                        # diffusion entre les bk dans les deux sens + metaplasticité sur le dernier                                
                            if bk_idx==(n_bk-1):
                                condition = (state['bk'+str(bk_idx-1)+'_t-1'] - state['bk'+str(bk_idx)+'_t-1'])*state['bk'+str(bk_idx)+'_t-1'] < 0
                                decayed_m = 1 - torch.tanh(meta[p.newname]*state['bk'+str(bk_idx)+'_t-1'])**2
                                state['bk'+str(bk_idx)+'_t'] = torch.where(condition, state['bk'+str(bk_idx)+'_t-1'] + (ratios[bk_idx-1]/areas[bk_idx])*decayed_m*(state['bk'+str(bk_idx-1)+'_t-1'] - state['bk'+str(bk_idx)+'_t-1']) + (ratios[bk_idx]/areas[bk_idx])*(state['bk'+str(bk_idx+1)+'_t-1'] - state['bk'+str(bk_idx)+'_t-1']), 
                                                                                      state['bk'+str(bk_idx)+'_t-1'] + (ratios[bk_idx-1]/areas[bk_idx])*(state['bk'+str(bk_idx-1)+'_t-1'] - state['bk'+str(bk_idx)+'_t-1']) + (ratios[bk_idx]/areas[bk_idx])*(state['bk'+str(bk_idx+1)+'_t-1'] - state['bk'+str(bk_idx)+'_t-1']))
                            else:
                                state['bk'+str(bk_idx)+'_t'] = state['bk'+str(bk_idx)+'_t-1'] + (ratios[bk_idx-1]/areas[bk_idx])*(state['bk'+str(bk_idx-1)+'_t-1'] - state['bk'+str(bk_idx)+'_t-1']) + (ratios[bk_idx]/areas[bk_idx])*(state['bk'+str(bk_idx+1)+'_t-1'] - state['bk'+str(bk_idx)+'_t-1'])
                    

                # Plotting beaker levels and distributions
                fig = plt.figure(figsize=(12,9))
                for bk_idx in range(n_bk):
                    if bk_idx==0:
                        state['bk'+str(bk_idx)+'_t-1'] = p.data
                    else:
                        state['bk'+str(bk_idx)+'_t-1'] = state['bk'+str(bk_idx)+'_t'] 
                    
                    if p.size() == torch.empty(4096,4096).size() :
                        state['bk'+str(bk_idx)+'_lvl'].append(state['bk'+str(bk_idx)+'_t-1'][11, 100].detach().item())
                        if state['step']%600==0:
                            plt.plot(state['bk'+str(bk_idx)+'_lvl'])
                            fig.savefig(path + '/trajectory.png', fmt='png', dpi=300)
                plt.close()
                
                if p.dim()!=1 and state['step']%600==0:
                    fig2 = plt.figure(figsize=(12,9))
                    for bk_idx in range(n_bk):
                        plt.hist(state['bk'+str(bk_idx)+'_t-1'].detach().cpu().numpy().flatten(), 100, label='bk'+str(bk_idx), alpha=0.5)
                    plt.legend()
                    fig2.savefig(path+'/bk_'+str(bk_idx)+'_'+str(p.size(0))+'-'+str(p.size(1))+'_task'+str((state['step']//48000)%2)+'.png', fmt='png')
                    torch.save(state, path + '/state_'+str(p.size(0))+'-'+str(p.size(1))+'_task'+str((state['step']//48000)%2)+'.tar')
                    plt.close()   
                
                
        return loss  