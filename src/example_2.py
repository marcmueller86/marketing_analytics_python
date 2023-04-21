import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

import IPython
IPython.embed()
# Load whisky data
whisky = pd.read_csv('Data/whisky.csv')

# Load world data
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Filter for UK data
UK = world[world['name'] == 'United Kingdom']

# Convert whiskies data to a DataFrame
whiskies_coord = pd.DataFrame({'Distillery': whisky['Distillery'],
                               'Latitude': whisky['Latitude'],
                               'Longitude': whisky['Longitude']})
# Convert whisky DataFrame to a GeoDataFrame
whiskies_gdf = gpd.GeoDataFrame(whiskies_coord, geometry=gpd.points_from_xy(whiskies_coord.Longitude, whiskies_coord.Latitude))
# Set Coordinate Reference System (CRS)
whiskies_gdf.crs = "EPSG:27700"
whiskies_gdf = whiskies_gdf.to_crs("EPSG:4326")
# Filter for Scotland data
Scotland = UK.dissolve()
# Plot Scotland map
# fig, ax = plt.subplots()
# Scotland.boundary.plot(ax=ax, linewidth=1, edgecolor='black', facecolor='white')
# whiskies_gdf.plot(ax=ax, marker='o', color='red', markersize=5, alpha=.9)
# plt.show()

#We will now select all the tasting notes and, along with the names of the distilleries, store them in a separate DataFrame.

whisky_tastingnotes = whisky.loc[:, 'Distillery':'Floral']
# Gather the data from columns Body to Floral into a long format
whisky_score = whisky.melt(id_vars=['RowID', 'Distillery', 'Postcode', 'Latitude', 'Longitude'],
                           var_name='Review.point', value_name='Score',
                           col_level=0, value_vars=['Body', 'Sweetness', 'Smoky', 'Medicinal', 'Tobacco', 'Honey', 'Spicy', 'Winey', 'Nutty', 'Malty', 'Fruity', 'Floral'])

# Display the first few rows of the whisky_score DataFrame
whisky_score.head()

# plotting

import seaborn as sns
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load whisky data

# Gather the data from columns Body to Floral into a long format
whisky_score = whisky.melt(id_vars=['RowID', 'Distillery', 'Postcode', 'Latitude', 'Longitude'],
                           var_name='Review.point', value_name='Score',
                           col_level=0, value_vars=['Body', 'Sweetness', 'Smoky', 'Medicinal', 'Tobacco', 'Honey', 'Spicy', 'Winey', 'Nutty', 'Malty', 'Fruity', 'Floral'])

# Plot the bar chart with a facet for each distillery
g = sns.FacetGrid(whisky_score, col='Distillery', col_wrap=4, sharey=True)
g.map_dataframe(sns.barplot, x='Review.point', y='Score', ci=None, palette='viridis')
g.set_axis_labels("", "Score").set_titles("{col_name}")

# Remove x-axis ticks and labels
for ax in g.axes.flat:
    ax.set_xticks([])
    ax.set_xlabel('')

# Adjust space between subplots
g.fig.subplots_adjust(wspace=.2, hspace=.6)

# Display the plot
plt.show()


# kmeans clustinger
#elbow

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer

# Select columns from Sweetness to Floral
whisky_tastingnotes = whisky.loc[:, 'Sweetness':'Floral']

# Calculate the within-cluster sum of squares (WSS) for different numbers of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(whisky_tastingnotes)
    wcss.append(kmeans.inertia_)

# Plot the elbow method graph
plt.plot(range(1, 11), wcss)
plt.title("Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WSS")
plt.show()

#silhoulette
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from yellowbrick.cluster import SilhouetteVisualizer
import numpy as np
from collections import Counter


# Select columns from Sweetness to Floral
whisky_tastingnotes = whisky.loc[:, 'Sweetness':'Floral']

# Determine the optimal number of clusters based on the silhouette method
silhouette_scores = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    cluster_labels = kmeans.fit_predict(whisky_tastingnotes)
    silhouette_avg = silhouette_score(whisky_tastingnotes, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Plot the silhouette method graph
plt.plot(range(2, 11), silhouette_scores)
plt.title("Silhouette Method")
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette Score")
plt.show()

# Visualize the silhouette plot for the optimal number of clusters
optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick')
visualizer.fit(whisky_tastingnotes)
visualizer.show()
# kmeans calc
# Select columns from Sweetness to Floral
whisky_tastingnotes = whisky.loc[:, 'Sweetness':'Floral']
# Set random seed for reproducibility
np.random.seed(123)
# Perform KMeans clustering with 4 clusters and 20 initializations
kmeans = KMeans(n_clusters=4, n_init=20, random_state=123)
whisky4cluster = kmeans.fit(whisky_tastingnotes)
# Get the cluster labels
cluster_labels = whisky4cluster.labels_
# Count the number of data points in each cluster
cluster_sizes = Counter(cluster_labels)
print(cluster_sizes)
# Get the cluster centers
cluster_centers = whisky4cluster.cluster_centers_
# Convert the cluster centers array to a DataFrame
cluster_centers_df = pd.DataFrame(cluster_centers, columns=whisky_tastingnotes.columns)
print(cluster_centers_df)


# euclidean distance

from scipy.spatial.distance import pdist, squareform

# Select columns from Sweetness to Floral and rows for the first 8 distilleries
tastingnotes_short = whisky.loc[whisky['Distillery'].isin(whisky['Distillery'][:8]), 'Sweetness':'Floral']

# Set the distillery names as row names
tastingnotes_short.index = whisky.loc[whisky['Distillery'].isin(whisky['Distillery'][:8]), 'Distillery']

# Compute the distance matrix
distance_matrix = squareform(pdist(tastingnotes_short))

print(distance_matrix)

#multi dimensonal
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform

# Select columns from Sweetness to Floral and rows for the first 20 distilleries
tastingnotes_longer = whisky.loc[whisky['Distillery'].isin(whisky['Distillery'][:20]), 'Sweetness':'Floral']

# Set the distillery names as row names
tastingnotes_longer.index = whisky.loc[whisky['Distillery'].isin(whisky['Distillery'][:20]), 'Distillery']

# Compute the distance matrix
d = pdist(tastingnotes_longer)
distance_matrix = squareform(d)

# Perform MDS with 2 dimensions
mds = MDS(n_components=2, dissimilarity='precomputed')
fit = mds.fit_transform(distance_matrix)

# Plot MDS
x = fit[:, 0]
y = fit[:, 1]
plt.figure(figsize=(12, 8))
plt.scatter(x, y, s=100)
for i, name in enumerate(tastingnotes_longer.index):
    plt.text(x[i], y[i], name, fontsize=12)
plt.xlabel("MDS dimension 1")
plt.ylabel("MDS dimension 2")
plt.title("Whisky MDS")
plt.show()

