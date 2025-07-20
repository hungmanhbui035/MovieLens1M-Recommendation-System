import argparse
import os
import wandb

import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_utils import download_ml1m, data_split
from models import GMF, MLP, NeuMF
from train_test_utils import train, validate, EarlyStopper, epoch_log

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sort", type=bool, default=False)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--model", type=str, default="gmf", choices=["gmf", "mlp", "neumf"])
    parser.add_argument("--emb-dim", type=int, default=32)
    parser.add_argument("--bias", type=bool, default=True)
    parser.add_argument("--layers", nargs='+', type=int, default=[64, 32, 16])
    parser.add_argument("--dropouts", nargs='+', type=float, default=[0.3, 0.4])
    parser.add_argument("--batch-norm", type=bool, default=True)
    parser.add_argument("--ckpt-path", type=str, default=None)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.3)
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--cosine-annealing", action="store_true")

    parser.add_argument("--log-freq", type=int, default=10)
    return parser.parse_args()

def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    download_ml1m()

    ratings_df = pd.read_csv("./ml-1m/ratings.dat", sep="::", header=None, names=["user_id", "movie_id", "rating", "timestamp"], engine="python")
    train_set, val_set, _, num_users, num_items, _, _ = data_split(ratings_df, test_ratio=0.2, sort=args.sort)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    if args.model == "gmf":
        model = GMF(num_users, num_items, args.emb_dim, args.bias).to(device)
    elif args.model == "mlp":
        model = MLP(num_users, num_items, args.layers, args.dropouts, args.batch_norm).to(device)
    elif args.model == "neumf":
        model = NeuMF(num_users, num_items, args.emb_dim, args.layers, args.dropouts, args.batch_norm).to(device)
    else:
        raise ValueError(f"Invalid model: {args.model}")
    model = nn.DataParallel(model)

    if not os.path.exists('./models'):
        os.makedirs('./models')
    model_path = f'./models/{args.model}.pth'

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.cosine_annealing:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    else:
        scheduler = None
    if args.early_stopping:
        early_stopper = EarlyStopper(model, model_path, patience=args.patience, min_delta=args.min_delta)
    else:
        early_stopper = None

    if args.ckpt_path:
        ckpt = torch.load(args.ckpt_path, map_location=device)
        model.module.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if scheduler and ckpt['scheduler']:
            scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
    else:
        start_epoch = 1

    wandb.login()
    wandb.init(project='MovieLens1M-Recommendation-System')
    wandb.config.update(args)
    wandb.watch(model, log="gradients", log_freq=args.log_freq)
    
    if not os.path.exists('./ckpts'):
        os.makedirs('./ckpts')
    for epoch in range(start_epoch, args.num_epochs + 1):
        train_loss = train(epoch, model, train_loader, criterion, optimizer, scheduler, device, args.log_freq)
        val_loss = validate(epoch, model, val_loader, criterion, device)
        epoch_log(epoch, train_loss, val_loss, args.num_epochs)

        if early_stopper and early_stopper.early_stop(val_loss):
            print('Early stop at epoch', epoch)
            break

        if epoch % 50 == 0:
            ckpt = {
                'epoch': epoch,
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
            }
            ckpt_path = f'./ckpts/{args.model}_{epoch}.pth'
            torch.save(ckpt, ckpt_path)
            artifact = wandb.Artifact('ckpt', type='model')
            artifact.add_file(ckpt_path)
            wandb.log_artifact(artifact)

    if not early_stopper:
        torch.save(model.module.state_dict(), model_path)
    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

    wandb.finish()

if __name__ == "__main__":
    main()