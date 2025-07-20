import urllib.request
import zipfile
import os

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset, Dataset

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
    df["rating"] = df["rating"] - 3 # normalize rating to [-2, 2]

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

    return train_df, val_df, test_df

def mf_data_preprocess(train_df, val_df, test_df):
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

def get_features(users_df, movies_df):
    # user_features
    user_index_by_id = {id: idx for idx, id in enumerate(users_df["user_id"]) }
    gender_index_by_name = {"M":0, "F": 1}
    age_index_by_name = {1: 0, 18: 1, 25: 2, 35:3, 45: 4, 50: 5, 56:6}
    occupations = [
        "other",
        "academic/educator",
        "artist",
        "clerical/admin",
        "college/grad student",
        "customer service",
        "doctor/health care",
        "executive/managerial",
        "farmer",
        "homemaker",
        "K-12 student",
        "lawyer",
        "programmer",
        "retired",
        "sales/marketing",
        "scientist",
        "self-employed",
        "technician/engineer",
        "tradesman/craftsman",
        "unemployed",
        "writer"
    ]
    occupation_index_by_name = {name: index for index, name in enumerate(occupations)}

    num_users = len(users_df)
    gender_offset = num_users
    age_offset = gender_offset + len(gender_index_by_name)
    occupation_offset = age_offset + len(age_index_by_name)

    user_features = []
    for i in range(num_users):
        gender_index = gender_index_by_name[users_df["gender"][i]] + gender_offset
        age_index = age_index_by_name[users_df["age"][i]] + age_offset
        occupation_index = users_df["occupation"][i] + occupation_offset
        user_features.append([i, gender_index, age_index, occupation_index])
        
    # movie_features
    movie_index_by_id = {id: idx for idx, id in enumerate(movies_df["movie_id"])}
    genres = [
        "Action",
        "Adventure",
        "Animation",
        "Children's",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
    ]
    genre_index_by_name = {name:i for i, name in enumerate(genres)}
    
    num_movies = len(movies_df)
    movie_offset = occupation_offset + len(occupation_index_by_name)
    genre_offset = movie_offset + num_movies

    movie_features = []
    for i, movie_genres in enumerate(movies_df["genres"]):
        movie_feature = [movie_offset + i]
        for genre in movie_genres.split("|"):
            genre_index = genre_index_by_name[genre] + genre_offset
            movie_feature.append(genre_index)
        movie_features.append(movie_feature)
    
    total_inputs = genre_offset + len(genres)

    return user_index_by_id, movie_index_by_id, user_features, movie_features, total_inputs, num_users, num_movies, len(genres)


class FMDataset(Dataset):
    def __init__(self, rating_df, num_genres, user_index_by_id, movie_index_by_id, user_features, movie_features, total_inputs):
        self.rating_df = rating_df
        self.max_size = 5 + num_genres # 4 for len(user_features) + 1 formovie_index
        self.user_index_by_id = user_index_by_id
        self.movie_index_by_id = movie_index_by_id
        self.user_features = user_features
        self.movie_features = movie_features
        self.total_inputs = total_inputs

    def __len__(self):
        return len(self.rating_df)

    def __getitem__(self, i):
        user_index = self.user_index_by_id[self.rating_df["user_id"].iloc[i]]
        movie_index = self.movie_index_by_id[self.rating_df["movie_id"].iloc[i]]
        rating = self.rating_df["rating"].iloc[i]
        user_feature = self.user_features[user_index]
        movie_feature = self.movie_features[movie_index]
        padding_size = self.max_size - len(user_feature) - len(movie_feature)
        feature = user_feature + movie_feature + [self.total_inputs] * padding_size
        return torch.LongTensor(feature), rating



