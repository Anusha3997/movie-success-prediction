import pandas as pd
import numpy as np
import ast

# Load the cleaned dataset
df = pd.read_csv("data/cleaned/cleaned.csv")

#Director Features
def add_director_features(df):

#Convert crew column (JSON) and get director
    def extract_directors(x):
        try:
            crew = ast.literal_eval(x)
            return [d['name'] for d in crew if d.get('job') == 'Director']
        except:
            return []

    df['director_list'] = df['crew'].apply(extract_directors)

    #Identifying the top directors and creating the feature columns
    major_directors = [
        "Christopher Nolan",
        "Steven Spielberg",
        "James Cameron",
        "Quentin Tarantino",
        "Ridley Scott",
        "Peter Jackson",
        "David Fincher",
        "Martin Scorsese",
        "Denis Villeneuve",
        "Jon Favreau",
        "J.J. Abrams",
        "Zack Snyder"
    ]
    for d in major_directors:
        clean_d = d.replace(" ", "_").replace(".", "")
        df[f'director_{clean_d}'] = df['director_list'].apply(lambda x: 1 if d in x else 0)
    df['director_other'] = df['director_list'].apply(
        lambda x: 1 if (len(x) > 0 and x[0] not in major_directors) else 0
    )
    return df

#Company Features
def add_company_features(df):
    #Extracting the company list from the production_companies column
    def extract_company_names(x):
        try:
            companies = ast.literal_eval(x)
            return [d['name'] for d in companies]
        except:
            return []
            
    df['company_list'] = df['production_companies'].apply(extract_company_names)

    #Identifying major companies and creating feature columns
    major_companies = [
        "Walt Disney Pictures",
        "Warner Bros.",
        "Universal Pictures",
        "Columbia Pictures",
        "Paramount Pictures",
        "20th Century Fox",
        "Marvel Studios",
        "Pixar",
        "Lionsgate",
        "MGM",
        "DreamWorks Animation",
        "New Line Cinema",
        "Legendary Pictures"
    ]
    for company in major_companies:
        clean_name = company.replace(" ", "_").replace(".", "")
        df[f'company_{clean_name}'] = df['company_list'].apply(lambda x: 1 if company in x else 0)
    return df

#Genre Features
def add_genre_features(df):
    
#Extracting the genre list from the genre column
    def extract_genre_names(x):
        try:
            genres = ast.literal_eval(x)
            return [d['name'] for d in genres]
        except:
            return []

    df['genre_list'] = df['genres'].apply(extract_genre_names)

    major_genres = [
        "Drama",
        "Comedy",
        "Action",
        "Thriller",
        "Romance",
        "Adventure",
        "Horror"
    ]
    #Identifying major genres and creating feature column
    for g in major_genres:
        clean_g = g.replace(" ", "_")
        df[f'genre_{clean_g}'] = df['genre_list'].apply(lambda x: 1 if g in x else 0)
    return df


def create_features():

    df = pd.read_csv("data/cleaned/cleaned.csv")

    df = add_director_features(df)
    df = add_company_features(df)
    df = add_genre_features(df)
    #drop non feature and non label columns
    df = df.drop(['genres', 'production_companies', 'production_countries', 'cast', 'crew',
    'company_list', 'genre_list', 'director_list'],axis = 1)
    #Converting the bool columns to int
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    df.to_csv("data/features/features.csv", index=False)
    print("✅ Features data saved -> features.csv")
    
if __name__ == "__main__":
    create_features()