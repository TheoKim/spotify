#%% 0) Init - import packages, set up data

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kstest

import matplotlib.pyplot as plt 
from scipy.stats import chi2_contingency

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA 


import random 
mySeed = 11063254
random.seed(mySeed)

songDataPandas = pd.read_csv('newSpotifyDataset2.csv')

Columns_to_read = [
    'songNumber', 'artists', 'album_name', 'track_name', 'popularity',
    'duration', 'explicit', 'danceability', 'energy', 'key', 'loudness',
    'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo', 'time_signature', 'track_genre'
]

#%% 1) Determining what categories (if any) are normally distributed
numBins = 100

Hist_categories = [
    'duration', 'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]

plt.hist(songDataPandas['popularity'], bins=numBins)
plt.xlabel('Popularity')
plt.ylabel('Count')
plt.title("Popularity Histogram")
plt.show()
    
for category in Hist_categories:
    plt.hist(songDataPandas[category], bins=numBins)
    plt.xlabel(category.capitalize())
    plt.ylabel('Count')
    plt.title(f"{category.capitalize()} Histogram")
    plt.show()
    
# Run the KS statistic and p-value
        
ks_statistic, p_value = kstest(songDataPandas['danceability'], 'norm')

print(f"KS Statistic: {ks_statistic}")
print(f"P-value: {p_value}")

ks_statistic2, p_value2 = kstest(songDataPandas['tempo'], 'norm')

print(f"KS Statistic: {ks_statistic2}")
print(f"P-value: {p_value2}")

#%% 2) Calculating the popularity of each type of music

genre_median_popularity = songDataPandas.groupby('track_genre')['popularity'].median().sort_values()

# Step 2: Calculate the overall median popularity
median_popularity = songDataPandas['popularity'].median()

# Step 3: Plot the bar chart for median popularity of each genre
plt.figure(figsize=(20, 8))
genre_median_popularity.plot(kind='bar', color='skyblue')
plt.xlabel('Genre')
plt.ylabel('Median Popularity')
plt.title('Median Popularity of Music Genres')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

shameArray = genre_median_popularity[genre_median_popularity < 10].index.tolist()
print('Genres with a median popularity less than 10:')
print(shameArray)

# Identifying genres with median popularity greater than 50
prideArray = genre_median_popularity[genre_median_popularity > 50].index.tolist()
print('Genres with a median popularity greater than 50:')
print(prideArray)

# Calculating the number of songs with a popularity of 0
num_songs_popularity_zero = songDataPandas[songDataPandas['popularity'] == 0].shape[0]
print(f'The number of songs with a popularity of 0 is: {num_songs_popularity_zero}')

#%% 2.5) chi-squared test

median_popularity = songDataPandas['popularity'].median()

songDataPandas['popularity_category'] = np.where(songDataPandas['popularity'] < median_popularity, 'Below Median', 'Above Median')

contingency_table = pd.crosstab(songDataPandas['track_genre'], songDataPandas['popularity_category'])

categories_chi2, categories_p, categores_dof, categories_expected = chi2_contingency(contingency_table)

print(f"Chi-squared Statistic: {categories_chi2}")
print(f"P-value: {categories_p}")
print(f"Degrees of Freedom: {categores_dof}")

categories_alpha = 0.05
print(f"With {categores_dof} degrees of freedom, a Chi-squared score of {categories_chi2}, and a p-value of {categories_p}, there is a significant difference between categories of music.")

# I looked up a critical value table for the above conclusion.

genre_median_popularity = songDataPandas.groupby('track_genre')['popularity'].median().sort_values()

median_popularity = songDataPandas['popularity'].median()

plt.figure(figsize=(20, 8))
genre_median_popularity.plot(kind='bar', color='skyblue')
plt.xlabel('Genre')
plt.ylabel('Median Popularity')
plt.axhline(y=median_popularity, color='red', linestyle='--', label=f'Overall Median Popularity = {median_popularity:.0f}')
plt.title('Median Popularity of Music Genres')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.legend()
plt.show()

    
#%% 3) Correlation between popularity and song attributes

correlation_results = {}

for category in Hist_categories:
    correlation = songDataPandas['popularity'].corr(songDataPandas[category])
    correlation_results[category] = correlation

fig, corrPLOT = plt.subplots(figsize=(10,7))

categories = list(correlation_results.keys())
correlations = list(correlation_results.values())

bars = corrPLOT.bar(categories, correlations, color='skyblue')

for bar, correlation in zip(bars, correlations):
    height = bar.get_height()
    corrPLOT.annotate(f'{correlation:.4f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

plt.xticks(rotation=45, ha='right')
corrPLOT.set_xlabel('Category')
corrPLOT.set_ylabel('Correlation with Popularity')
corrPLOT.set_title('Correlation between Popularity and Features')
plt.tight_layout()
plt.show()


#%% 4) Is there a significant difference between the popularity of explicit and clean songs?

# Filter explicit and non-explicit songs into separate DataFrames
explicitSongs = songDataPandas[songDataPandas['explicit']]
explicitNotsongs = songDataPandas[~songDataPandas['explicit']]

popularityMean = songDataPandas['explicit'].mean()

# Create dfs for explicit and non-explicit (clean) popularity ratings
popularityExplicit = pd.DataFrame({'Popularity of explicit songs': explicitSongs['popularity']})
popularityNotExplicit = pd.DataFrame({'Popularity of not explicit songs': explicitNotsongs['popularity']})

medianExplicit = np.median(popularityExplicit)
medianClean = np.median(popularityNotExplicit)

plt.hist(popularityExplicit)
plt.xlabel("Popularity")
plt.axvline(medianExplicit, color='red', linestyle='dashed', linewidth=1)
plt.ylabel("Count")
plt.title('Popularity of Explicit Songs')
plt.show()

plt.hist(popularityNotExplicit)
plt.axvline(medianClean, color='red', linestyle='dashed', linewidth=1)
plt.xlabel("Popularity")
plt.ylabel("Count")
plt.title('Popularity of Clean Songs')
plt.show()

popularityExplicit = popularityExplicit['Popularity of explicit songs'].dropna()
popularityNotExplicit = popularityNotExplicit['Popularity of not explicit songs'].dropna()

# Run Mann-Whitney U Test
u1,p1 = stats.mannwhitneyu(popularityExplicit,popularityNotExplicit)

print("Explicit vs. clean songs popularity")
print("MW Value: ", u1)
print("p-value: ", p1)

print("Explicit songs median: ", medianExplicit)
print("Clean songs median: ", medianClean)


#%% 5) Is there a significant difference between the popularity of songs in major or minor key?

majorSongs = songDataPandas[songDataPandas['mode'] == 1]
minorsongs = songDataPandas[songDataPandas['mode'] == 0]

popularityMajor = pd.DataFrame({'Popularity of major songs': majorSongs['popularity']})
popularityMinor = pd.DataFrame({'Popularity of minor songs': minorsongs['popularity']})

medianMajor = np.median(popularityMajor)
medianMinor = np.median(popularityMinor)

plt.hist(popularityMajor)
plt.axvline(medianMajor, color='red', linestyle='dashed', linewidth=1)
plt.xlabel("Popularity")
plt.ylabel("Count")
plt.title('Popularity of Major Songs')
plt.show()

plt.hist(popularityMinor)
plt.axvline(medianMinor, color='red', linestyle='dashed', linewidth=1)
plt.xlabel("Popularity")
plt.ylabel("Count")
plt.title('Popularity of Minor Songs')
plt.show()

popularityMajor = popularityMajor.dropna()
popularityMinor = popularityMinor.dropna()

u2,p2 = stats.mannwhitneyu(popularityMajor, popularityMinor)

print("Major vs. minor songs popularity")
print("MW Value: ", u2)
print("p-value: ", p2)

print("Major songs median: ", medianMajor)
print("Minor songs median: ", medianMinor)


#%% 6) Of 10 variables, finding the best linear regression predictor of popularity. 

model = LinearRegression()
max_r_squared = 0
max_category = ''

# Loop through the features for regression
for category in Hist_categories:

    X = songDataPandas[[category]].values
    y = songDataPandas['popularity'].values
    
    model.fit(X, y)
    y_pred = model.predict(X)
    r_squared = r2_score(y, y_pred)
    
    # Update max R-squared value and corresponding category if greater
    if r_squared > max_r_squared:
        max_r_squared = r_squared
        max_category = category

print(f"Greatest R-squared category: '{max_category}' R-squared of {max_r_squared}")

plt.figure(figsize=(10, 6))
plt.scatter(songDataPandas[max_category], songDataPandas['popularity'], label='Actual')
plt.plot(songDataPandas[max_category], y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel(max_category)
plt.ylabel('Popularity')
plt.title('Popularity Regression Plot')
plt.legend()
plt.show()

# The takeaway: We can't use one variable to predict popularity!

#%% 7) Multiple regression predicting popularity

# Does using multiple variables help us better predict popularity?

Xmult = songDataPandas[Hist_categories].values
Ymult = songDataPandas['popularity'].values

multModel = LinearRegression().fit(Xmult,Ymult)

rSqrFull = multModel.score(Xmult,Ymult)

b0, b1 = multModel.intercept_, multModel.coef_

YmultPrediction = multModel.predict(Xmult)

print(f"R-squared for multiple regression using all 10 categories: {rSqrFull}")

print("Difference in R-squared values: ", rSqrFull - max_r_squared)

plt.figure(figsize=(10, 6))
plt.scatter(YmultPrediction, Ymult)
pred_range = np.linspace(max(YmultPrediction.min(), 0), YmultPrediction.max(), 100)
best_fit_model = LinearRegression().fit(YmultPrediction.reshape(-1, 1), Ymult)
best_fit_line = best_fit_model.predict(pred_range.reshape(-1, 1))
plt.plot(pred_range, best_fit_line, color = 'red')
plt.xlabel('Predicted Popularity')
plt.ylabel('Actual Popularity')
plt.title('Multiple Regression: Actual vs Predicted Popularity')
plt.show()

#%% 8) Linear regression of energy and loudness

plt.figure(figsize=(10, 6))
plt.scatter(songDataPandas['energy'], songDataPandas['loudness'])
plt.xlabel('Energy')
plt.ylabel('decibels (dB)')
plt.title('Energy and loudness correlation')

model = LinearRegression()
X = songDataPandas[['energy']]
y = songDataPandas['loudness']
model.fit(X, y)
plt.plot(X, model.predict(X), color='red')

plt.show()

corrEnergyLoudness = np.corrcoef(songDataPandas['energy'], songDataPandas['loudness'])[0, 1]
print("Correlation between Energy and Loudness:", corrEnergyLoudness)


#%% 9) Principal component analysis

# Let's identify what concepts are the most important measures of popularity.
# Dimension reduction into a few, highly relevant "independent" variables.

songDataSubset = songDataPandas[Hist_categories]

corrMatrix = np.corrcoef(songDataSubset,rowvar=False)

# Plot the data:
plt.imshow(corrMatrix) 
plt.xlabel('Criterion')
plt.ylabel('Criterion')
plt.colorbar()
plt.show()

zscoredData = stats.zscore(songDataSubset)

# Perform PCA
pca = PCA().fit(zscoredData)

eigVals = pca.explained_variance_
loadings = pca.components_ 
rotatedData = pca.fit_transform(zscoredData)
numFeatures = len(Columns_to_read)

numComponents = len(eigVals)

x = np.linspace(0, numComponents-1, numComponents)

plt.bar(x, eigVals, color='gray')
plt.plot([0, numComponents], [1, 1], color='orange')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.title('Eigenvalues of Principal Components')
plt.show()

varExplained = eigVals/sum(eigVals)*100

# Elbow criterion: 1 component

# Kaiser criterion: 3 components

threshold = 90
eigSum = np.cumsum(varExplained)
print('Number of factors to account for at least 90% variance:', np.count_nonzero(eigSum < threshold) + 1)

for whichPrincipalComponent in range(1, 4):
    plt.bar(x, loadings[whichPrincipalComponent, :] * -1)
    plt.xlabel('Column')
    plt.ylabel('Loading')
    plt.title(f'Loading for Principal Component {whichPrincipalComponent}')
    plt.show()

var3total = varExplained[0] + varExplained[1] + varExplained[2]
print('3 principal components variance total: ' , var3total)

# PC 1: points towards danceability and valence, away from duration.
# The presence of "excitement" affects a song's popularity. Don't make it boring (too long!)
# Interpretation: How "exciting" a song is.

# PC 2: points towards from speechiness and liveness
# Presence of lyrics and liveness affects the popularity.
# Interpretation: The lyricism of a song

# PC 3: points towards tempo and points away from duration.
# Presence of tempo affects popularity.
# Absence of duration affects popularity.
# Interpretation: The "time feeling" of a song. How fast is the song? Does it overstay its welcome (too long)?

# These are the 3 dimensions/axes that determine popularity.

#%% 10) Visualizing data compared to popularity and in new axes

# Visualize the data in the new coordinate system against popularity

# PC1 vs Popularity
plt.figure(figsize=(12, 8))
plt.scatter(rotatedData[:, 0], songDataPandas['popularity'], alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Popularity')
plt.title('Popularity vs. Principal Component 1')
plt.tight_layout()
plt.show()

# PC2 vs Popularity
plt.figure(figsize=(12, 8))
plt.scatter(rotatedData[:, 1], songDataPandas['popularity'], alpha=0.5)
plt.xlabel('Principal Component 2')
plt.ylabel('Popularity')
plt.title('Popularity vs. Principal Component 2')
plt.tight_layout()
plt.show()

# PC3 vs Popularity
plt.figure(figsize=(12, 8))
plt.scatter(rotatedData[:, 2], songDataPandas['popularity'], alpha=0.5)
plt.xlabel('Principal Component 3')
plt.ylabel('Popularity')
plt.title('Popularity vs. Principal Component 3')
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

pc1 = rotatedData[:, 0]
pc2 = rotatedData[:, 1]
pc3 = rotatedData[:, 2]

ax.scatter(pc1, pc2, pc3, c='b', marker='o')

ax.set_xlabel('Principal Component 1: "Excitment"')
ax.set_ylabel('Principal Component 2: "Lyricism"')
ax.set_zlabel('Principal Component 3: "Time-sense"')
ax.set_title('3D Plot of First Three Principal Components')

plt.tight_layout()

plt.show()
