"""Point d'entrée en ligne de commande.

    python src/main.py prepare        # télécharge + prétraite le dataset (cache data/processed/)
"""

import argparse

from load_data import prepare_dataset
from utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare"])
    parser.add_argument("--config", default="baseline")
    parser.add_argument("--force", action="store_true", help="refaire le cache")
    args = parser.parse_args()
    cfg = load_config(args.config)["project"]

    if args.command == "prepare":
        data = prepare_dataset(cfg["image_size"], cfg["val_size"], cfg["seed"], force=args.force)
        s = data.stats
        print(f"train={s['n_train']}  val={s['n_val']}  test={s['n_test']}  "
              f"(retirés : {s['n_duplicates_removed']} doublons, "
              f"{s['n_near_test_removed']} copies d'images du test, "
              f"{s['n_near_duplicates_removed']} quasi-doublons)")
        print(f"mean={s['mean']:.4f}  std={s['std']:.4f}  taille={s['image_size']}px")


if __name__ == "__main__":
    main()
