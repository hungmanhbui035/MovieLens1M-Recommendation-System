import argparse

import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_utils import download_ml1m, data_split
from models import GMF, MLP, NeuMF
from train_test_utils import test

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gmf", choices=["gmf", "mlp", "neumf"])
    parser.add_argument("--bias", type=bool, default=True)
    parser.add_argument("--emb-dim", type=int, default=32)
    parser.add_argument("--layers", nargs='+', type=int, default=[64, 32, 16])
    parser.add_argument("--dropouts", nargs='+', type=float, default=[0.3, 0.4])
    parser.add_argument("--batch-norm", type=bool, default=True)
    parser.add_argument("--model-path", type=str, required=True)

    parser.add_argument("--sort", type=bool, default=False)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)

    return parser.parse_args()

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    download_ml1m()

    ratings_df = pd.read_csv("./ml-1m/ratings.dat", sep="::", header=None, names=["user_id", "movie_id", "rating", "timestamp"], engine="python")
    _, _, test_set, num_users, num_items, _, _ = data_split(ratings_df, test_ratio=0.2, sort=args.sort)

    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    if args.model == "gmf":
        model = GMF(num_users, num_items, args.emb_dim, args.bias).to(device)
    elif args.model == "mlp":
        model = MLP(num_users, num_items, args.layers, args.dropouts, args.batch_norm).to(device)
    elif args.model == "neumf":
        model = NeuMF(num_users, num_items, args.emb_dim, args.layers, args.dropouts, args.batch_norm).to(device)
    else:
        raise ValueError(f"Invalid model: {args.model}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = nn.DataParallel(model)

    criterion = nn.MSELoss()

    test_loss = test(model, loader, criterion, device)

    print(f"test loss {test_loss:.4f}")

if __name__ == "__main__":
    main()