import torch

class SharedRMSprop(torch.optim.RMSprop):
    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        super().__init__(params, lr, alpha, eps, weight_decay, momentum, centered)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                
                state['square_avg'] = torch.zeros_like(p.data)
                state['square_avg'].share_memory_()
                
                if momentum > 0:
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                    state['momentum_buffer'].share_memory_()
                
                if centered:
                    state['grad_avg'] = torch.zeros_like(p.data)
                    state['grad_avg'].share_memory_()