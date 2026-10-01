import json
import random
import sys
import types
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms
from resnet_model import PreActResNet18

HERE = Path(__file__).resolve().parent
ASSETS = next((p for p in (HERE, HERE.parents[1] / "hw1")
               if (p / "models").is_dir()), HERE)
CACHE = HERE / "uar_reference"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
BASE = "https://raw.githubusercontent.com/ddkang/advex-uar/fd9e0efa1155cedb6d15942ed6e22e1bf1252043/advex_uar/"
FAMILIES = ["jpeg_linf", "jpeg_l2", "jpeg_l1", "elastic"]
MODEL_IDS = {"Linf": "1IjYkXHj5XAF_9U5QW1MlXwiEbRPqUq5H",
             "L2": "1jC8sFpVXpQrmRVxqxhA3skcl768ro_w5",
             "RAMP": "1XFsvaOzgmEFWABN_dk1UhNv0fBJXOxHf"}


def download(name, url):
    CACHE.mkdir(exist_ok=True)
    path = CACHE / name
    if not path.exists():
        with urllib.request.urlopen(url, timeout=60) as response:
            path.write_bytes(response.read())
    return path


def load_reference():
    package = "_cs521_uar_reference"
    for name in [package, package + ".attacks"]:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    modules = {}
    for name in ["attacks", "elastic", "jpeg", "jpeg_attack", "elastic_attack"]:
        path = download(name + ".py", BASE + "attacks/" + name + ".py")
        source = path.read_text().replace("from advex_uar.attacks.", f"from {package}.attacks.")
 
        source = source.replace("device='cuda'", f"device='{DEVICE}'")
        source = source.replace('device="cuda"', f'device="{DEVICE}"')
        source = source.replace(".cuda()", f'.to("{DEVICE}")')
        source = source.replace("l1norms > base_eps,", "l1norms > base_eps[:, None],")
        source = source.replace("F.grid_sample(im, flow, mode='bilinear')",
                                "F.grid_sample(im, flow, mode='bilinear', align_corners=True)")
        module = types.ModuleType(f"{package}.attacks.{name}")
        sys.modules[module.__name__] = module
        exec(compile(source, str(path), "exec"), module.__dict__)
        modules[name] = module
    path = download("calibration.json", BASE + "analysis/calibrations/cifar-10/calibs.out")
    calibration = [c for c in json.loads(path.read_text()) if c[0][0] in FAMILIES]
    assert all(sum(c[0][0] == family for c in calibration) == 6 for family in FAMILIES)
    return modules, calibration


class BenchmarkModel(nn.Module):
    def __init__(self, raw_model):
        super().__init__()
        self.raw_model = raw_model
        self.register_buffer("mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, normalized):
        return self.raw_model(normalized * self.std + self.mean)


def compute_uar(calibration, accuracies):
    result = {}
    for family in sorted(FAMILIES):
        cells = [cell for cell in calibration if cell[0][0] == family]
        result[family] = sum(accuracies[(family, float(c[0][1]))] for c in cells) / sum(c[1] for c in cells)
    return result


def evaluate(name, images, labels, modules, calibration):
    checkpoint = ASSETS / "models" / f"pretr_{name}.pth"
    if not checkpoint.exists():
        import gdown
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(id=MODEL_IDS[name], output=str(checkpoint), quiet=False)
    model = PreActResNet18(10, cuda=False, activation="softplus1")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model = model.to(DEVICE).eval().requires_grad_(False)
    model.normalize.mu = model.normalize.mu.to(DEVICE)
    model.normalize.std = model.normalize.std.to(DEVICE)
    adapter = BenchmarkModel(model).to(DEVICE).eval()
    batch_size = 50 if name == "RAMP" else 100
    clean = []
    for start in range(0, len(images), batch_size):
        x, y = images[start:start + batch_size].to(DEVICE), labels[start:start + batch_size].to(DEVICE)
        with torch.no_grad():
            clean.extend(model(x).argmax(1).eq(y).cpu().tolist())
    accuracies = {}
    for cell_index, cell in enumerate(calibration):
        (family, epsilon, steps, step_size), _, _ = cell
        if family.startswith("jpeg_"):
            attack = modules["jpeg_attack"].JPEGAttack(steps, epsilon, step_size, 32, opt=family[5:])
        else:
            attack = modules["elastic_attack"].ElasticAttack(steps, epsilon, step_size, 32)
        batches = []
        for start in range(0, len(images), batch_size):
            x, y = images[start:start + batch_size].to(DEVICE), labels[start:start + batch_size].to(DEVICE)
            normalized = (x - adapter.mean) / adapter.std
            survived = torch.tensor(clean[start:start + len(x)], device=DEVICE)
            seed = 42 + cell_index * 100000 + start * 10
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            adv = attack(adapter, normalized, y, avoid_target=True, scale_eps=False).detach()
            with torch.no_grad():
                raw = adv * adapter.std + adapter.mean
                assert torch.isfinite(raw).all() and raw.min() >= -1e-5 and raw.max() <= 1 + 1e-5
                survived &= adapter(adv).argmax(1).eq(y)
            batches.append(survived.cpu().numpy())
        accuracies[family, float(epsilon)] = float(np.concatenate(batches).mean())
        print(f"{name}: {family} eps={epsilon:g}, {100 * accuracies[family, float(epsilon)]:.2f}%", flush=True)
    return {"clean_accuracy": float(np.mean(clean)), "uar": compute_uar(calibration, accuracies)}


def main():
    torch.set_num_threads(2)
    modules, calibration = load_reference()
    data = datasets.CIFAR10(ASSETS / "cifar10_data", train=False, download=True,
                           transform=transforms.ToTensor())
    rng = np.random.default_rng(42)
    groups = [rng.permutation(np.flatnonzero(np.asarray(data.targets) == c)) for c in range(10)]
    ids = [groups[i % 10][i // 10] for i in range(100)]
    images = torch.stack([data[int(i)][0] for i in ids])
    labels = torch.tensor([data.targets[int(i)] for i in ids])
    for name in MODEL_IDS:
        print(json.dumps({name: evaluate(name, images, labels, modules, calibration)}, indent=2))


if __name__ == "__main__":
    main()
