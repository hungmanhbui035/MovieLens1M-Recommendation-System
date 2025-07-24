import argparse

import pandas as pd

import torch

from data_utils import download_ml1m, data_split, get_features, get_not_seen_movies
from models import FM, NeuFM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sort", type=bool, default=True)
    parser.add_argument("--model", type=str, default="fm", choices=["fm", "neufm"])
    parser.add_argument("--emb-dim", type=int, default=64)
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
    users_df = pd.read_csv("./ml-1m/users.dat", sep="::", header=None, names=["user_id", "gender", "age", "occupation", "zip_code"], engine="python")
    movies_df = pd.read_csv("./ml-1m/movies.dat", sep="::", header=None, names=["movie_id", "title", "genres"], engine="python", encoding="latin-1")
    train_df, val_df, test_df = data_split(ratings_df, test_ratio=0.2, sort=args.sort)
    user_index_by_id, movie_index_by_id, user_features, movie_features, total_inputs, num_users, num_movies, num_genres = get_features(users_df, movies_df)
    index_to_id_movie = {idx: id for id, idx in movie_index_by_id.items()}

    if args.model == "fm":
        model = FM(total_inputs, args.emb_dim).to(device)
    elif args.model == "neufm":
        model = NeuFM(total_inputs, args.layers, args.dropouts, args.batch_norm).to(device)
    else:
        raise ValueError(f"Invalid model: {args.model}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    print(f"\nThere are {num_users} users in the dataset")
    try:
        user_id = int(input("Enter user id: "))
        user_idx = user_index_by_id[user_id]
    except KeyError:
        print("This user_id does not exist")
        user_id = int(input("Enter user id: "))
        user_idx = user_index_by_id[user_id]
    except ValueError:
        print("user_id must be an integer starting from 1")
        user_id = int(input("Enter user id: "))
        user_idx = user_index_by_id[user_id]

    seen_movies, not_seen_movies = get_not_seen_movies(user_id, train_df, val_df, test_df)
    print(f"\nUser {user_id} has seen {len(seen_movies)} movies, and has not seen {len(not_seen_movies)} movies")
    print(f"\nSeen movies:")
    for movie_id in seen_movies:
        movie_title = movies_df[movies_df["movie_id"] == movie_id]["title"].values[0]
        print(f"- {movie_title}")

    user_feature = user_features[user_idx]
    features = []
    for movie_id in not_seen_movies:
        movie_idx = movie_index_by_id[movie_id]
        movie_feature = movie_features[movie_idx]
        feature = user_feature + movie_feature + [total_inputs] * (5 + num_genres - len(user_feature) - len(movie_feature))
        features.append(feature)
    input_tensor = torch.tensor(features, dtype=torch.long)
    
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    top_k_indices = torch.topk(output, k=10)[1].tolist()

    print(f"\nTop 10 movies recommended for user {user_id}:")
    for i in top_k_indices:
        movie_id = index_to_id_movie[i]
        movie_title = movies_df[movies_df["movie_id"] == movie_id]["title"].values[0]
        print(f"- {movie_title}")

if __name__ == "__main__":
    main()