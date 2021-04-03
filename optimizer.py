import torch


def unitwise_norm(x, norm_type=2.0):
    if x.ndim <= 1:
        return x.norm(norm_type)

    else:
        return x.norm(norm_type, dim=tuple(range(1, x.ndim)), keepdim=True)


@torch.no_grad()
def adaptive_grad_clip(parameters, clipping=0.01, eps=1e-3):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    for p in parameters:
        if p.grad is None:
            continue

        grad = p.grad
        max_norm = unitwise_norm(p).clamp_(min=eps).mul_(clipping)
        g_norm = unitwise_norm(grad)
        clipped_grad = grad * (max_norm / g_norm.clamp(min=1e-6))
        new_grad = torch.where(g_norm < max_norm, grad, clipped_grad)
        p.grad.copy_(new_grad)
