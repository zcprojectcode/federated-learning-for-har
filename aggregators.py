"""
MIT License

Copyright (c) 2026 Farhad Rezazadeh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch.optim as optim

class FedAvgOptimizer(optim.AdamW):
    """
    Standard AdamW optimiser
    """
    pass

class FedNovaOptimizer(optim.AdamW):
    """
    Standard AdamW optimiser
    """
    pass

class FedPerOptimizer(optim.AdamW):
    """
    Standard AdamW optimiser
    """
    pass

class FedProxOptimizer(optim.AdamW):
    """
    AdamW with proximal term L2 regularisation
    """
    def __init__(self, params, lr, mu):
        super().__init__(params, lr=lr)
        self.mu = mu

    def step(self, closure=None, global_params=None):
        if global_params is None:
            return super().step(closure)

        # Add proximal term
        for p, g in zip(self.param_groups[0]['params'], global_params.parameters()):
            if p.grad is not None:
                p.grad.data.add_(self.mu * (p.data - g.data))

        # Step with corrected gradients
        return super().step(closure)

class FedDANEOptimizer(optim.AdamW):
    """
    Same as FedProx with gradient correction by global gradient snapshot
    """
    def __init__(self, params, lr, mu):
        super().__init__(params, lr=lr)
        self.mu = mu

    def step(self, closure=None, global_params=None, global_gradients=None):
        if global_params is None or global_gradients is None:
            return super().step(closure)

        # Collect local gradients 
        local_grads = [
            p.grad.data.clone() if p.grad is not None else torch.zeros_like(p.data)
            for p in self.param_groups[0]['params']
        ]

        # Apply DANE correction to gradients before AdamW
        for p, g, gg, lg in zip(
            self.param_groups[0]['params'],
            global_params.parameters(),
            global_gradients,
            local_grads
        ):
            if p.grad is not None:
                p.grad.data = lg + gg.data + self.mu * (p.data - g.data)

        # Step with corrected gradients
        loss = super().step(closure)
        return loss

class FedSGDOptimizer(optim.Optimizer):
    """
    SGD optimiser. One local epoch for each client
    """
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def step(self, closure=None, global_gradients=None):
        loss = None
        if closure is not None:
            loss = closure()
        if global_gradients is None:
            return loss
        for group in self.param_groups:
            for p, g in zip(group['params'], global_gradients):
                p.data.add_(g.data, alpha=-group['lr'])
        return loss