from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from resnet_model import PreActResNet18

STEPS = 20  # The question leaves k unspecified.
EPS_LINF, EPS_L2 = 8 / 255, 0.75
DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
HERE = Path(__file__).resolve().parent
ASSETS = next((p for p in (HERE, HERE.parents[1] / "hw1")
               if (p / "models").is_dir()), HERE)
MODEL_IDS = {"Linf": "1IjYkXHj5XAF_9U5QW1MlXwiEbRPqUq5H",
             "L2": "1jC8sFpVXpQrmRVxqxhA3skcl768ro_w5",
             "RAMP": "1XFsvaOzgmEFWABN_dk1UhNv0fBJXOxHf"}


def _check(k, eps, eps_step):
    if k < 0 or eps < 0 or eps_step < 0:
        raise ValueError("Steps, radius, and step size must be nonnegative.")


@torch.enable_grad()
def pgd_linf_untargeted(model, x, labels, k, eps, eps_step):
    _check(k, eps, eps_step)
    model.eval()
    x0 = x.detach()
    adv_x = x0.clone()
    for _ in range(k):
        adv_x.requires_grad_(True)
        loss = F.cross_entropy(model(adv_x), labels)
        grad = torch.autograd.grad(loss, adv_x)[0]
        with torch.no_grad():
            adv_x = adv_x + eps_step * grad.sign()
            delta = (adv_x - x0).clamp(-eps, eps)
            adv_x = (x0 + delta).clamp(0.0, 1.0)
    return adv_x.detach()


@torch.enable_grad()
def pgd_l2_untargeted(model, x, labels, k, eps, eps_step):
    _check(k, eps, eps_step)
    model.eval()
    x0 = x.detach()
    adv_x = x0.clone()
    if eps == 0:
        return adv_x
    shape = (-1,) + (1,) * (x.ndim - 1)
    for _ in range(k):
        adv_x.requires_grad_(True)
        loss = F.cross_entropy(model(adv_x), labels)
        grad = torch.autograd.grad(loss, adv_x)[0]
        with torch.no_grad():
            grad_norm = grad.flatten(1).norm(p=2, dim=1).view(shape)
            adv_x = adv_x + eps_step * grad / (grad_norm + 1e-10)
            delta = adv_x - x0
            delta_norm = delta.flatten(1).norm(p=2, dim=1).view(shape)
            delta = eps * delta / delta_norm.clamp_min(eps)
            adv_x = (x0 + delta).clamp(0.0, 1.0)
    return adv_x.detach()


def load_models():
    torch.manual_seed(42)
    data = datasets.CIFAR10(ASSETS / "cifar10_data", train=False, download=True,
                           transform=transforms.ToTensor())
    loader = DataLoader(data, batch_size=256, shuffle=False, num_workers=0)
    for name, file_id in MODEL_IDS.items():
        checkpoint = ASSETS / "models" / f"pretr_{name}.pth"
        if not checkpoint.exists():
            import gdown
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            gdown.download(id=file_id, output=str(checkpoint), quiet=False)
        model = PreActResNet18(10, cuda=False, activation="softplus1")
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        model = model.to(DEVICE).eval().requires_grad_(False)
        model.normalize.mu = model.normalize.mu.to(DEVICE)
        model.normalize.std = model.normalize.std.to(DEVICE)
        print(f"Evaluating {name}: {len(data)} images, {STEPS} steps, {DEVICE}", flush=True)
        yield name, model, loader


def evaluate_model(model, loader):
    correct = [[], [], []]
    for i, (x, labels) in enumerate(loader):
        x, labels = x.to(DEVICE), labels.to(DEVICE)
        with torch.no_grad():
            pred_clean = model(x).argmax(1)
        adv_linf = pgd_linf_untargeted(model, x, labels, STEPS, EPS_LINF, EPS_LINF / 4)
        with torch.no_grad():
            pred_linf = model(adv_linf).argmax(1)
        adv_l2 = pgd_l2_untargeted(model, x, labels, STEPS, EPS_L2, EPS_L2 / 4)
        with torch.no_grad():
            pred_l2 = model(adv_l2).argmax(1)
        for parts, pred in zip(correct, (pred_clean, pred_linf, pred_l2)):
            parts.append(pred.eq(labels).cpu())
        if i % 5 == 0 or i == len(loader) - 1:
            print(f"  {min((i + 1) * loader.batch_size, len(loader.dataset))}/{len(loader.dataset)} images", flush=True)
    return tuple(torch.cat(parts) for parts in correct)


def main():
    for name, model, loader in load_models():
        clean, linf, l2 = evaluate_model(model, loader)
        a = [100 * mask.sum().item() / len(mask) for mask in (clean, linf, l2)]
        print(f"{name}: Standard={a[0]:.2f}%, Linf PGD={a[1]:.2f}%, L2 PGD={a[2]:.2f}%")


if __name__ == "__main__":
    main()
