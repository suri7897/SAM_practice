import torch


class MultiSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, n_step=2, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(MultiSAM, self).__init__(params, defaults)

        self.n_step = n_step
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        grad_norm = self._grad_norm()
        n_steps = self.n_step
        rho_step = self.param_groups[0]["rho"] / float(n_steps)

        for group in self.param_groups:
            scale = rho_step / (grad_norm + 1e-12)
            adaptive = group["adaptive"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if "old_p" not in self.state[p]:
                    self.state[p]["old_p"] = p.data.clone()
                e = (torch.pow(p, 2) if adaptive else 1.0) * p.grad * scale.to(p)
                p.add_(e)  # w <- w + e

        if zero_grad:
            self.zero_grad(set_to_none=True)


    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                old = self.state[p].pop("old_p", None)
                if old is not None:
                    p.data = old  # return to w

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad(set_to_none=True)



    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        n_steps = self.n_step

        self.zero_grad(set_to_none=True)
        closure()

        for _ in range(n_steps - 1):
            self.first_step(zero_grad=True)  
            closure()                 

        self.first_step(zero_grad=True)      # w <- w + e_n

        closure()

        self.second_step(zero_grad=True)

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
