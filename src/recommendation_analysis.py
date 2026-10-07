import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
MOVIES_DATA_PATH = "data/movies.csv"
RATINGS_DATA_PATH = "data/ratings.csv"
movies_df=pd.read_csv(MOVIES_DATA_PATH)
ratings_df=pd.read_csv(RATINGS_DATA_PATH)
print("Movies dataset:",movies_df.head())
print("\n Ratings dataset:",ratings_df.head())
print("\n Movies dataset shape:",movies_df.shape)
print("\n Ratings datset shape:",ratings_df.shape)
print("\n Movies dataset information")
movies_df.info()
print("\n Ratings dataset information")
ratings_df.info()
print("\n Missing values in movies dataset")
print(movies_df.isnull().sum())
print("\n Missing values in ratings dataset")
print(ratings_df.isnull().sum())
print("\n Duplicate rows in movie dataset")
print(movies_df.duplicated().sum())
print("\n Dplicate rows in ratings dataset")
print(ratings_df.duplicated().sum())
print("\n Number of unique users:")
print(ratings_df["userId"].nunique())
print("\n Number of unique movies in ratinsg dataset")
print(ratings_df["movieId"].nunique())
print("\nRatings statistics")
print(ratings_df["rating"].describe())
print("\n Rating value counts:")
print(ratings_df["rating"].value_counts().sort_index())
movie_rating_stats=ratings_df.groupby("movieId").agg(ratings_count=("rating","count"),average_rating=("rating","mean")).reset_index()
print("\nMovie rating statistics")
print(movie_rating_stats)
movie_popularity_df=movies_df.merge(movie_rating_stats,on="movieId",how="inner")
print("\nMovie popularity data:")
print(movie_popularity_df.head())
minimum_ratings=movie_popularity_df["ratings_count"].quantile(0.90)
print("\n Minimum ratings count required")
print(minimum_ratings)
popular_movies_df=movie_popularity_df[movie_popularity_df["ratings_count"]>=minimum_ratings].copy()
print(popular_movies_df)
popular_movies_df=popular_movies_df.sort_values(by=["average_rating","ratings_count"],ascending=[False,False])
print("\n Top 10 popular movies")
print(popular_movies_df[["title","average_rating","ratings_count"]].head(10))
tfidf_vectorizer=TfidfVectorizer(token_pattern=r"[^|]+")
genre_matrix=tfidf_vectorizer.fit_transform(movies_df["genres"])
print("\n Genre feature matrix shape:")
print(genre_matrix.shape)
print("\n Genre features")
print(tfidf_vectorizer.get_feature_names_out())
#genre_similarity_matrix=cosine_similarity(genre_matrix)
#print("\n genre similarity matrix shape")
#print(genre_similarity_matrix.shape)
movie_indices=pd.Series(movies_df.index,index=movies_df["title"])
print("\n Movie title to index mapping")
print(movie_indices.head())
selected_movie="Toy Story (1995)"
movie_index=movie_indices[selected_movie]
print("\n selected movie")
print(selected_movie)
print("\n Selected movie index")
print(movie_index)
movie_similarity_scores=cosine_similarity(genre_matrix[movie_index],genre_matrix).ravel()
print("\n selected movie similarity score shape")
print(movie_similarity_scores.shape)
#movie_similarity_scores=genre_similarity_matrix[movie_index]
#print("\n similairty scores for selected movie")
print(movie_similarity_scores[:10])
movie_similarity_df=pd.DataFrame({"movie_index":range(len(movie_similarity_scores)),
                                  "similarity_score":movie_similarity_scores})
print("\nMovie similarity scores with indices")
print(movie_similarity_df)
movie_similarity_df=movie_similarity_df.sort_values(by=["similarity_score","movie_index"],ascending=[False,True])
print("\n Movies sorted by similarity")
print(movie_similarity_df.head(10))
movie_similarity_df=movie_similarity_df[movie_similarity_df["movie_index"]!=movie_index].copy()
print("\n Similair movies excluding selected movies")
top_similar_movies_df=movie_similarity_df.head(10)
print("\n Top similiar movie indices and scores")
print(top_similar_movies_df)
print("\n Movies with highest genre similarity")
print(movies_df.iloc[top_similar_movies_df["movie_index"].to_numpy()][["title","genres"]])
recommended_movies_df=movies_df.iloc[top_similar_movies_df["movie_index"].to_numpy()][["movieId","title","genres"]].copy()
print("Recommended movie details")
print(recommended_movies_df)
recommended_movies_df["similiarity_score"]=top_similar_movies_df["similarity_score"]
recommended_movies_df=recommended_movies_df.reset_index(drop=True)
print("\n Final content based movie recommendations")
print(recommended_movies_df)
movie_user_matrix=ratings_df.pivot_table(index="movieId",columns="userId",values="rating")
print("\n Movie user rating matrix shape")
print(movie_user_matrix.iloc[:5,:5])
print(movie_user_matrix.shape)
selected_movie_id=1
selected_movie_ratings=movie_user_matrix.loc[selected_movie_id]
print("\n selected movie ratings")
print(selected_movie_ratings.head(10))
print("\n Number of users who rated the selected movie")
print(selected_movie_ratings.count())
comparision_movie_id=2
comparision_movie_ratings=movie_user_matrix.loc[comparision_movie_id]
common_ratings_df=pd.DataFrame({"selected_movie_ratings":selected_movie_ratings,
              "comparision_movie_rating":comparision_movie_ratings}).dropna()
print("\n Users who rated both movies")
print(common_ratings_df.head(10))
print(len(common_ratings_df))
selected_movie_vector=common_ratings_df[["selected_movie_ratings"]].T
comparision_movie_vector=common_ratings_df[["comparision_movie_rating"]].T
movie_similarity_score=cosine_similarity(selected_movie_vector,comparision_movie_vector)[0][0]
print("\n cosine similarity between the two movies")
print(movie_similarity_score)
min_common_users=20
common_users_count=len(common_ratings_df)
print("Number of common users")
print(common_users_count)
if common_users_count>=min_common_users:
    print(f"\n sufficient common users")
    print("cosine similarity:",movie_similarity_score)
else:
    print("\n Insufficient common users")
    print("similarity result wont be used")
movie_similarity_results=[]
for comparision_movie_id in movie_user_matrix.index:
    if comparision_movie_id==selected_movie_id:
        continue
    comparision_movie_ratings=movie_user_matrix.loc[comparision_movie_id]
    common_ratings_df=pd.DataFrame({"selected_movie_ratings":selected_movie_ratings,
                  "comparision_movie_ratings":comparision_movie_ratings}).dropna()
    common_users_count=len(common_ratings_df)
    if common_users_count<min_common_users:
        continue
    #selected_movie_vector=common_ratings_df[["selected_movie_ratings"]].T
    #comparision_movie_vector=common_ratings_df[["comparision_movie_ratings"]].T
    #movie_similarity_score=cosine_similarity(selected_movie_vector,comparision_movie_vector)[0][0]
    movie_similarity_score=common_ratings_df["selected_movie_ratings"].corr(common_ratings_df["comparision_movie_ratings"])
    if pd.isna(movie_similarity_score):
        continue
    movie_similarity_results.append({"movieId":comparision_movie_id,
                                     "similarity_score":movie_similarity_score,
                                     "common_users_count":common_users_count})
collaborative_similarity_df=pd.DataFrame(movie_similarity_results)
print("\n Collaborative filtering simlarity results")
print(collaborative_similarity_df)
print("\n Number of eligible comparision movies")
print(len(collaborative_similarity_df))
collaborative_similarity_df=collaborative_similarity_df.sort_values(by=["similarity_score","common_users_count"],
                                                                    ascending=[False,False])
print("\n collaborative similarity results after sorting")
print(collaborative_similarity_df.head(10))
top_collaborative_movies_df=collaborative_similarity_df.head(10)
print("\n Top 10 collaborative filtering movie similarities")
print(top_collaborative_movies_df)
collaborative_recommendations_df=top_collaborative_movies_df.merge(movies_df[["movieId","title","genres"]],
                                                                   on="movieId",how="left")
print("\n Final collaborative filtering recommendations")
print(collaborative_recommendations_df)
#comparision_movie_id=2
#comparision_movie_ratings=movie_user_matrix.loc[comparision_movie_id]
#pearson_common_ratings_df=pd.DataFrame({"selected_movie_ratings":selected_movie_ratings,"comparision_movie_ratings":comparision_movie_ratings}).dropna()
#pearson_similarity_score=pearson_common_ratings_df["selected_movie_ratings"].corr(pearson_common_ratings_df["comparision_movie_ratings"])
#print("\n Number of common users for pearson comparision")
#print(len(pearson_common_ratings_df))
#print("\n Pearson similarity between the two movies")
#print(pearson_similarity_score)
print(ratings_df.head(10))
user_movie_matrix=ratings_df.pivot(index="userId",columns="movieId",values="rating")
print("\nUser movie rating matrix")
print(type(user_movie_matrix))
print(user_movie_matrix.iloc[:5,:5])
print("\n User movie rating matrix shape")
print(user_movie_matrix.shape)
selected_user_id=1
selected_user_ratings=user_movie_matrix.loc[selected_user_id]
print("\n selected user ratings")
print(selected_user_ratings.head(10))
print("\n Number of movies rated by selected user id")
print(selected_user_ratings.notna().sum())
comparision_user_id=2
comparision_user_ratings=user_movie_matrix.loc[comparision_user_id]
common_user_ratings_df=pd.DataFrame({"selected_user_ratings":selected_user_ratings,
                                     "comparision_user_ratings":comparision_user_ratings}).dropna()
print("\n Movies rated by both users")
print(common_user_ratings_df.head(10))
print("\n Number of movies rated by both users")
print(len(common_user_ratings_df))
min_common_movies=5
user_similarity_results=[]
for comparision_user_id in user_movie_matrix.index:
    if comparision_user_id==selected_user_id:
        continue
    comparision_user_ratings=user_movie_matrix.loc[comparision_user_id]
    common_user_ratings_df=pd.DataFrame({"selected_user_ratings":selected_user_ratings,
                                         "comparision_user_ratings":comparision_user_ratings}).dropna()
    common_movies_count=len(common_user_ratings_df)
    if common_movies_count<min_common_movies:
        continue
    if (common_user_ratings_df["selected_user_ratings"].nunique()<=1
       or 
       common_user_ratings_df["comparision_user_ratings"].nunique()<=1):
           continue
    user_similarity_score=common_user_ratings_df["selected_user_ratings"].corr(common_user_ratings_df["comparision_user_ratings"])
    if pd.isna(user_similarity_score):
        continue
    user_similarity_results.append({"userId":comparision_user_id,"similarity_score":user_similarity_score,
                                    "common_movies_count":common_movies_count})
user_similarity_df=pd.DataFrame(user_similarity_results)
print("\n User similarity results")
print(user_similarity_df)
print("\n Number of eligible similiar users")
print(len(user_similarity_df.head(10)))
user_similarity_df=user_similarity_df.sort_values(by=["similarity_score","common_movies_count"],ascending=[False,False])
print("\n Users similarity results after sorting")
print(user_similarity_df)
positive_user_similarity_df=user_similarity_df[user_similarity_df["similarity_score"]>0]
top_similiar_users_df=positive_user_similarity_df.head(10)
print("Top 10 positive similiar users")
print(top_similiar_users_df)
selected_user_watched_movie_ids=selected_user_ratings[selected_user_ratings.notna()].index
print("\n Movie IDs already rated by selected user")
print(selected_user_watched_movie_ids)
print("\n Number of movies already rated by selected user")
print(len(selected_user_watched_movie_ids))
top_similiar_users_ids=top_similiar_users_df["userId"].to_list()
print("\n Top similiar user IDs")
print(top_similiar_users_ids)
similiar_user_ratings_df=ratings_df[ratings_df["userId"].isin(top_similiar_users_ids)]
print("\n Ratings given by top similiar users")
print(similiar_user_ratings_df.head(10))
print("\n Number of ratings given by top similiar users")
print(len(similiar_user_ratings_df))
candidate_movie_ratings_df=similiar_user_ratings_df[~similiar_user_ratings_df["movieId"].isin(selected_user_watched_movie_ids)]
print("\n Candidate movie ratings after removing already rated movies")
print("\n Number of candidate movie rating rows")
print(len(candidate_movie_ratings_df))
candidate_movie_ratings_df=candidate_movie_ratings_df.merge(top_similiar_users_df[["userId","similarity_score"]],
                                                            on="userId",
                                                            how="left")
print("\nCandidate movie ratings with similarity score")
print(candidate_movie_ratings_df[["userId","movieId","rating","similarity_score"]].head(10))
candidate_movie_ratings_df["weighted_rating"]=candidate_movie_ratings_df["rating"]*candidate_movie_ratings_df["similarity_score"]
print("\n Candidate movie ratings with weighted rating")
print(candidate_movie_ratings_df[
        [
            "userId",
            "movieId",
            "rating",
            "similarity_score",
            "weighted_rating"]].head(10))
movie_recommendation_scores_df=candidate_movie_ratings_df.groupby("movieId").agg(weighted_rating_sum=("weighted_rating","sum"),
                                                                                 similarity_sum=("similarity_score","sum"),
                                                                                 similar_users_count=("userId","count")).reset_index()
print("\n Movie recommendation score components")
print(movie_recommendation_scores_df.head(10))
movie_recommendation_scores_df["recommendation_score"]=movie_recommendation_scores_df["weighted_rating_sum"]/movie_recommendation_scores_df["similarity_sum"]
print("\nMovie recommendation scores")
print(movie_recommendation_scores_df[["movieId","recommendation_score","similar_users_count"]].head(10))
movie_recommendation_scores_df=movie_recommendation_scores_df.sort_values(by=["recommendation_score","similar_users_count"],
                                                                          ascending=[False,False])
print("\n Movie recommendation scores after sorting")
print(movie_recommendation_scores_df[["movieId","recommendation_score","similar_users_count"]].head(10))
top_recommended_movies_df=movie_recommendation_scores_df.head(10)
top_recommended_movies_df=top_recommended_movies_df.merge(movies_df[["movieId","title","genres"]],
                                                          on="movieId",
                                                          how="left")
print("\n Top 10 movie recommendations for selected user")
print(top_recommended_movies_df[["movieId","title","genres","recommendation_score","similar_users_count"]])
reliable_movie_recommendation_df=movie_recommendation_scores_df[movie_recommendation_scores_df["similar_users_count"]>=2]
print("\n Recommendations rated by atleast 2 users")
print(reliable_movie_recommendation_df[["movieId","recommendation_score","similar_users_count"]].head(10))
print("\n Number of relaible recommendation candidates")
print(len(reliable_movie_recommendation_df))
top_recommended_movies_df=reliable_movie_recommendation_df.head(10)
top_recommended_movies_df=top_recommended_movies_df.merge(movies_df[["movieId","title","genres"]],
                                                          on="movieId",
                                                          how="left")
print("\nFinal top 10 user-based movie recommendations")
print(
    top_recommended_movies_df[
        [
            "movieId",
            "title",
            "genres",
            "recommendation_score",
            "similar_users_count"
        ]
    ]
)
already_rated_recommendations=top_recommended_movies_df["movieId"].isin(selected_user_watched_movie_ids).sum()
print("\n Number of recommended movies already rated by selected user")
print(already_rated_recommendations)
#selected_user_ratings=user_movie_matrix.loc[selected_user_id]
if selected_user_id not in user_movie_matrix.index:
    print("\n Cold start user detected")
    print("\n No rating history available for this user")
    print("Use popularity based recommendations as fallback")
else:
    selected_user_ratings=user_movie_matrix.loc[selected_user_id]