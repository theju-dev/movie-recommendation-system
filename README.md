# Movie Recommendation System

## Project Overview

This project implements a movie recommendation system using the MovieLens dataset.

The objective is to explore multiple recommendation approaches and understand how movies can be recommended using overall popularity, movie characteristics, item-to-item relationships, and user-to-user rating behavior.

The project implements:

- Popularity-Based Recommendation
- Content-Based Recommendation
- Item-Based Collaborative Filtering
- User-Based Collaborative Filtering

## Dataset

The project uses the MovieLens dataset.

The main files used are:

- `movies.csv` — contains movie IDs, titles, and genres.
- `ratings.csv` — contains user IDs, movie IDs, ratings, and timestamps.

The ratings data contains approximately 100,000 user-movie ratings from 610 users.

## Recommendation Approaches

### 1. Popularity-Based Recommendation

The popularity-based recommender recommends movies using overall rating behavior rather than the preferences of a specific user.

This approach is useful when personalized user information is unavailable and can also act as a fallback for new users.

### 2. Content-Based Recommendation

The content-based recommender recommends movies that are similar to a selected movie based on movie genres.

Movie genres are transformed into numerical features using TF-IDF.

Cosine similarity is then used to measure the similarity between the selected movie and other movies.

The movies with the highest similarity scores are returned as recommendations.

### 3. Item-Based Collaborative Filtering

Item-based collaborative filtering recommends movies by comparing rating patterns between movies.

For a selected movie:

1. Users who rated both movies are identified.
2. Ratings from the common users are compared.
3. Pearson correlation is calculated between the two movies.
4. Movies with stronger positive correlations are ranked higher.

A minimum number of common users is used to avoid relying on correlations calculated from insufficient rating information.

### 4. User-Based Collaborative Filtering

User-based collaborative filtering recommends movies based on users with similar rating behavior.

For a selected user:

1. The user's existing movie ratings are identified.
2. Other users are compared with the selected user.
3. Only movies rated by both users are used for similarity calculation.
4. Pearson correlation is used to calculate user similarity.
5. Positively similar users are selected.
6. Movies already rated by the selected user are removed.
7. Candidate movies are scored using similarity-weighted ratings.
8. Movies are ranked to generate the final recommendations.

The recommendation score is calculated as a weighted average:

`Recommendation Score = Sum(Rating × User Similarity) / Sum(User Similarity)`

This gives users with stronger similarity more influence over the final recommendation score.

## Recommendation Reliability

A movie can receive a high recommendation score even when only one similar user has rated it.

To reduce reliance on such weak evidence, the project applies a minimum similar-user support threshold before generating the final recommendations.

When recommendation scores are equal, movies supported by more similar users are preferred.

## Already-Rated Movie Filtering

Movies already rated by the selected user are removed from the candidate recommendation set.

A final validation check confirms that the generated recommendations do not contain movies the selected user has already rated.

## Cold-Start Problem

User-based collaborative filtering requires existing user rating history.

A completely new user has no ratings, so user similarity cannot be calculated.

The project identifies this cold-start situation conceptually and uses popularity-based recommendations as the fallback strategy.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Cosine Similarity
- Pearson Correlation

## Project Structure

```text
movie_recommendation_system/
│
├── data/
│   ├── movies.csv
│   └── ratings.csv
│
├── src/
│   └── recommendation_analysis.py
│
├── models/
├── outputs/
├── .gitignore
├── requirements.txt
└── README.md
```

## Key Learning Outcomes

This project demonstrates:

- Building multiple recommendation strategies.
- Popularity-based recommendation.
- Feature representation using TF-IDF.
- Movie similarity using cosine similarity.
- Item-based collaborative filtering.
- User-based collaborative filtering.
- Pearson correlation for rating-pattern similarity.
- Similarity-weighted recommendation scoring.
- Filtering already-rated movies.
- Applying minimum support thresholds to improve recommendation reliability.
- Understanding the cold-start problem in recommendation systems.
- Translating recommendation-system concepts into practical Python code.

## Future Improvements

Possible future improvements include:

- Refactoring recommendation logic into reusable components.
- Separating training and prediction responsibilities.
- Adding structured logging and error handling.
- Adding automated tests.
- Exposing recommendations through an API.
- Containerizing and deploying the application.
```

One small note: if your `models/` and `outputs/` folders are currently empty, GitHub won't display them because **Git doesn't track empty directories**. That's completely normal.

Now save this into:

```text id="h2ngma"
README.md
```

Then run:

```bash id="n4y1ts"
git status
```

At this stage we should see the project files as untracked, while `.venv/` should still be absent.

**Don't `git add` or commit yet.** Once we verify that status, we'll establish the branch workflow correctly before the first commit.