import torch
from torch import nn

torch.manual_seed(13)
N = nn.Sequential(nn.Linear(10, 10, bias=False), nn.ReLU(),
                  nn.Linear(10, 10, bias=False), nn.ReLU(),
                  nn.Linear(10, 3, bias=False))
x = torch.rand(1, 10)
N.eval().requires_grad_(False)
t = 1
eps = 0.5 - 1e-7


def report(name, z):
    scores = N(z).detach()
    delta = z - x
    assert delta.abs().max().item() <= 0.5
    print(f"{name}: class={scores.argmax(1).item()}, "
          f"Linf={delta.abs().max().item():.9f}, L2={delta.norm().item():.9f}")
    print("  logits:", scores[0].tolist())


print("Original class:", N(x).argmax(1).item())

# Repeat targeted FGSM with t = 1; the prediction remains class 2.
z = x.clone().requires_grad_(True)
loss = nn.CrossEntropyLoss()(N(z), torch.tensor([t]))
grad = torch.autograd.grad(loss, z)[0]
fgsm_x = (z - eps * grad.sign()).detach()
report("FGSM target 1", fgsm_x)

adv_x = x.clone()
for step in range(1, 101):
    adv_x.requires_grad_(True)
    logits = N(adv_x)
    loss = logits[0, [0, 2]].max() - logits[0, t]
    grad = torch.autograd.grad(loss, adv_x)[0]
    with torch.no_grad():
        adv_x = adv_x - 0.1 * grad.sign()
        adv_x = x + (adv_x - x).clamp(-eps, eps)
        scores = N(adv_x)
        if scores[0, t] > scores[0, [0, 2]].max():
            break
else:
    raise RuntimeError("Target 1 was not reached")

assert N(adv_x).argmax(1).item() == t
report("Margin PGD target 1", adv_x)
print("Margin attack updates:", step)
print("Adversarial input:", adv_x[0].tolist())
