import urllib.request
import zipfile
import os

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset

def download_ml1m():
    if os.path.exists("./ml-1m"):
        print("Data already downloaded!")
        return

    URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    file_name = "ml-1m.zip"

    urllib.request.urlretrieve(URL, file_name)
    with zipfile.ZipFile(file_name, 'r') as z:
        z.extractall()

    os.remove(file_name)

    print("Data downloaded!")

def data_split(df, test_ratio, sort):
    train_ratio = 1 - 2 * test_ratio
    
    if sort:
        df = df.sort_values(by='timestamp')

        n = len(df)
        n1 = int(n * train_ratio)
        n2 = int(n1 + n * test_ratio)

        train_df = df.iloc[:n1]
        val_df = df.iloc[n1:n2]
        test_df = df.iloc[n2:]
    else:
        train_df, val_test_df = train_test_split(df, test_size=2*test_ratio, random_state=42)
        val_df, test_df = train_test_split(val_test_df, test_size=0.5, random_state=42)

    all_user_ids = train_df["user_id"].unique()
    all_item_ids = train_df["movie_id"].unique()
    num_users, num_items = len(all_user_ids), len(all_item_ids)

    val_df = val_df[val_df["user_id"].isin(all_user_ids) & val_df["movie_id"].isin(all_item_ids)]
    test_df = test_df[test_df["user_id"].isin(all_user_ids) & test_df["movie_id"].isin(all_item_ids)]

    user_id_mapping = {uid: idx for idx, uid in enumerate(sorted(all_user_ids))}
    item_id_mapping = {aid: idx for idx, aid in enumerate(sorted(all_item_ids))}

    for df in [train_df, val_df, test_df]:
        df['user_id'] = df['user_id'].map(user_id_mapping)
        df['movie_id'] = df['movie_id'].map(item_id_mapping)
    
    train_tensor = torch.tensor(train_df[['user_id', 'movie_id', 'rating']].values, dtype=torch.long)
    val_tensor = torch.tensor(val_df[['user_id', 'movie_id', 'rating']].values, dtype=torch.long)
    test_tensor = torch.tensor(test_df[['user_id', 'movie_id', 'rating']].values, dtype=torch.long)

    train_set = TensorDataset(train_tensor[:, :2], train_tensor[:, 2].to(torch.float32))
    val_set = TensorDataset(val_tensor[:, :2], val_tensor[:, 2].to(torch.float32))
    test_set = TensorDataset(test_tensor[:, :2], test_tensor[:, 2].to(torch.float32))

    return train_set, val_set, test_set, num_users, num_items, user_id_mapping, item_id_mapping