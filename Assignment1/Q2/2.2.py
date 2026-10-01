from pathlib import Path
import runpy

part1 = runpy.run_path(str(Path(__file__).with_name("2.1.py")))


def union_accuracy(correct_linf, correct_l2):
    return 100 * (correct_linf & correct_l2).sum().item() / len(correct_linf)


def main():
    for name, model, loader in part1["load_models"]():
        _, linf, l2 = part1["evaluate_model"](model, loader)
        print(f"{name}: Union={union_accuracy(linf, l2):.2f}%")


if __name__ == "__main__":
    main()
