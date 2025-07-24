import argparse

import pandas as pd

import torch

from data_utils import download_ml1m, data_split, mf_data_preprocess, get_not_seen_movies
from models import GMF, MLP, NeuMF


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sort", type=bool, default=False)
    parser.add_argument("--model", type=str, default="gmf", choices=["gmf", "mlp", "neumf"])
    parser.add_argument("--emb-dim", type=int, default=32)
    parser.add_argument("--bias", type=bool, default=True)
    parser.add_argument("--layers", nargs='+', type=int, default=[64, 32, 16])
    parser.add_argument("--dropouts", nargs='+', type=float, default=[0.3, 0.4])
    parser.add_argument("--batch-norm", type=bool, default=True)
    parser.add_argument("--model-path", type=str, required=True)

    return parser.parse_args()

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    download_ml1m()

    ratings_df = pd.read_csv("./ml-1m/ratings.dat", sep="::", header=None, names=["user_id", "movie_id", "rating", "timestamp"], engine="python")
    movies_df = pd.read_csv("./ml-1m/movies.dat", sep="::", header=None, names=["movie_id", "title", "genres"], engine="python", encoding="latin-1")
    train_df, val_df, test_df = data_split(ratings_df, test_ratio=0.2, sort=args.sort)
    train_df, val_df, test_df, _, _, _, num_users, num_items, user_id_mapping, item_id_mapping = mf_data_preprocess(train_df, val_df, test_df)
    index_to_id_item = {idx: id for id, idx in item_id_mapping.items()}

    if args.model == "gmf":
        model = GMF(num_users, num_items, args.emb_dim, args.bias).to(device)
    elif args.model == "mlp":
        model = MLP(num_users, num_items, args.layers, args.dropouts, args.batch_norm).to(device)
    elif args.model == "neumf":
        model = NeuMF(num_users, num_items, args.emb_dim, args.layers, args.dropouts, args.batch_norm).to(device)
    else:
        raise ValueError(f"Invalid model: {args.model}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    print(f"\nThere are {num_users} users in the dataset")
    try:
        
        user_id = int(input("Enter user id: "))
        user_idx = user_id_mapping[user_id]
    except KeyError:
        print("This user_id does not exist")
        user_id = int(input("Enter user id: "))
        user_idx = user_id_mapping[user_id]
    except ValueError:
        print("user_id must be an integer starting from 1")
        user_id = int(input("Enter user id: "))
        user_idx = user_id_mapping[user_id]

    seen_movies, not_seen_movies = get_not_seen_movies(user_idx, train_df, val_df, test_df)
    print(f"\nUser {user_id} has seen {len(seen_movies)} movies, and has not seen {len(not_seen_movies)} movies")
    print(f"\nSeen movies:")
    for movie_idx in seen_movies:
        movie_id = index_to_id_item[movie_idx]
        movie_title = movies_df[movies_df["movie_id"] == movie_id]["title"].values[0]
        print(f"- {movie_title}")

    user_tensor = torch.tensor([user_idx] * len(not_seen_movies), dtype=torch.long)
    movie_tensor = torch.tensor(not_seen_movies, dtype=torch.long)
    input_tensor = torch.stack([user_tensor, movie_tensor], dim=1)
    
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    top_k_indices = torch.topk(output, k=10)[1].tolist()

    print(f"\nTop 10 movies recommended for user {user_id}:")
    for i in top_k_indices:
        movie_id = index_to_id_item[i]
        movie_title = movies_df[movies_df["movie_id"] == movie_id]["title"].values[0]
        print(f"- {movie_title}")

if __name__ == "__main__":
    main()