import numpy as np
import pandas as pd
import IPython
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
# for correlations
from scipy.stats import pearsonr
import seaborn as sns
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, leaves_list
# kmeans
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
# kmeans counter center 
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer
# clusting hierachie
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cut_tree
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
# pca
from sklearn.decomposition import PCA


import IPython
IPython.embed()
df_satisfaction_1_xlsx = pd.read_excel('Data/satisfaction_1.xlsx', sheet_name='satisfaction_v2')
df_satisfaction_xlsx = pd.read_excel('Data/satisfaction.xlsx', sheet_name='Sheet1')


df_satisfaction_xlsx['Gender'] = df_satisfaction_xlsx['Gender'].astype('category')

# gender distribution
counts = df_satisfaction_xlsx['Gender'].value_counts()
# Create a bar plot
plt.bar(counts.index, counts.values)
plt.title('Female/Male Participants')
plt.xlabel('Gender')
plt.ylabel('Anzahl')
plt.show()

# satisfaction distribution
df_satisfaction_xlsx['Satisfaction'] = df_satisfaction_xlsx['Satisfaction'].astype('category')
counts_satisfaction = df_satisfaction_xlsx['Satisfaction'].value_counts()
# Create a bar plot
plt.bar(counts_satisfaction.index, counts_satisfaction.values)
plt.title('Zufriedene und unzufriedene Kunden')
plt.xlabel('Satisfaction')
plt.ylabel('Anzahl')
plt.show()


#histogram
# Assuming 'satisfaction' is a Pandas DataFrame
ages = df_satisfaction_xlsx['Age']
# Create a histogram
plt.hist(ages, edgecolor='black')
plt.title('Histogramm der Altersverteilung')
plt.xlabel('Alter in Jahren')
plt.ylabel('Anzahl')
plt.show()


# Assuming 'satisfaction' is a Pandas DataFrame
distances = df_satisfaction_xlsx['Disctance']  # Please note there's a typo in 'Disctance', it should be 'Distance'
# Create a histogram
plt.hist(distances, edgecolor='black')
plt.title('Histogramm der Flugdistanzen')
plt.xlabel('Distanz in Meilen')
plt.ylabel('Anzahl')
plt.show()

#density plot
distances = df_satisfaction_xlsx['Disctance']  # Please note there's a typo in 'Disctance', it should be 'Distance'
# Estimate the density of the distances
density = gaussian_kde(distances)
# Create a range of values for the x-axis
x = np.linspace(min(distances), max(distances), 1000)
# Create a density plot
plt.plot(x, density(x), color='blue')
plt.fill_between(x, density(x), color='red', alpha=0.5)
plt.title('Dichte der zurückgelegten Distanzen')
plt.xlabel('Distanz in Meilen')
plt.ylabel('Dichte')
plt.show()

# summary
distance_summary = distances = df_satisfaction_xlsx['Disctance'].describe()  
distance_summary

# histogram seats (7 ticks)
seat_comfort = df_satisfaction_xlsx['SeatComf']
# Create a histogram
plt.hist(seat_comfort, edgecolor='black', bins=range(1, 7))
plt.title('Sitzkomfort')
plt.xlabel('Rating')
plt.ylabel('Anzahl')
plt.xticks(range(1, 7))
plt.show()

# satisfaction with seats
time_convenient = df_satisfaction_xlsx['TimeConvenient']
# Create a histogram
plt.hist(time_convenient, edgecolor='black', bins=5)
plt.title('Zufriedenheit mit Abflugzeit')
plt.xlabel('Rating')
plt.ylabel('Anzahl')
plt.show()

# food and drinks satisfaction
food = df_satisfaction_xlsx['Food']
# Create a histogram
plt.hist(food, edgecolor='black', color='yellow', bins=5)
plt.title('Essen und Getränke')
plt.xlabel('Rating')
plt.ylabel('Anzahl')
plt.show()


# reaching the gate 
gate = df_satisfaction_xlsx['Gate']
# Create a histogram
plt.hist(gate, edgecolor='red', color='blue', bins=5)
plt.title('Erreichbarkeit des Gates')
plt.xlabel('Rating')
plt.ylabel('Anzahl')
plt.show()


# reaching the gate 
inflight_service = df_satisfaction_xlsx['InflightService']
# Create a histogram
plt.hist(inflight_service, edgecolor='black', color='red', bins=5)
plt.title('Wifi an Bord')
plt.xlabel('Rating')
plt.ylabel('Anzahl')
plt.show()


#entertainment 
entertainment = df_satisfaction_xlsx['Entertainment']
# Create a histogram
plt.hist(entertainment, edgecolor='black', color='red', bins=5)
plt.title('Unterhaltungssystem an Bord')
plt.xlabel('Rating')
plt.ylabel('Anzahl')
plt.show()

#online booking 
online_booking = df_satisfaction_xlsx['OnlineBooking']

# Create a histogram
plt.hist(online_booking, edgecolor='black', color='green', bins=5)
plt.title('Onlinebuchung')
plt.xlabel('Rating')
plt.ylabel('Anzahl')
plt.show()

# show correlations

selected_columns = ['SeatComf', 'TimeConvenient', 'Food', 'Gate', 'InflightService', 'Entertainment', 'Support', 'OnlineBooking', 'Service', 'LegRoom', 'Baggage', 'Checkin', 'Clean']
surveyquestions = df_satisfaction_xlsx[selected_columns]
surveyquestions.describe()
correlations_survey = surveyquestions.corr()
rounded_correlations = correlations_survey.round(2)

print(rounded_correlations)

# correlation matrix

# Assuming 'surveyquestions' is a Pandas DataFrame
correlations, p_values = [], []

for column1 in surveyquestions.columns:
    row_correlations, row_p_values = [], []
    for column2 in surveyquestions.columns:
        correlation, p_value = pearsonr(surveyquestions[column1], surveyquestions[column2])
        row_correlations.append(correlation)
        row_p_values.append(p_value)
    correlations.append(row_correlations)
    p_values.append(row_p_values)

correlations_matrix = pd.DataFrame(correlations, columns=surveyquestions.columns, index=surveyquestions.columns)
p_values_matrix = pd.DataFrame(p_values, columns=surveyquestions.columns, index=surveyquestions.columns)

# plot correlation ratios
print(correlations_matrix)
print(p_values_matrix)

distances = pdist(correlations_survey, metric='euclidean')
linkage_matrix = linkage(distances, method='ward')
cluster_order = leaves_list(linkage_matrix)

ordered_correlations = correlations_survey.iloc[cluster_order, cluster_order]

plt.figure(figsize=(10, 10))
sns.set(font_scale=0.8)
sns.heatmap(ordered_correlations, annot=True, cmap='cool', vmin=-1, vmax=1, square=True, linewidths=0.5, cbar=False)
plt.xticks(rotation=45, ha='right')
plt.title('Correlation Plot')
plt.show()

# correlations with p value

distances = pdist(correlations_matrix, metric='euclidean')
linkage_matrix = linkage(distances, method='ward')
cluster_order = leaves_list(linkage_matrix)

ordered_correlations = correlations_matrix.iloc[cluster_order, cluster_order]
ordered_p_values = p_values_matrix.iloc[cluster_order, cluster_order]

mask = np.zeros_like(ordered_correlations)
mask[np.triu_indices_from(mask)] = True
mask[ordered_p_values >= 0.01] = True

plt.figure(figsize=(10, 10))
sns.set(font_scale=0.8)
sns.heatmap(ordered_correlations, annot=True, cmap='cool', vmin=-1, vmax=1, square=True, linewidths=0.5, cbar=False, mask=mask)
plt.xticks(rotation=45, ha='right')
plt.title('Correlation Plot')
plt.show()

# kmeans 
# Assuming 'satisfaction' is a Pandas DataFrame
selected_columns = ['SeatComf', 'TimeConvenient', 'Food', 'Gate', 'InflightService', 'Entertainment', 'Support', 'OnlineBooking', 'Service', 'LegRoom', 'Baggage', 'Checkin', 'Clean', 'DelayDepart', 'DelayArrival', 'Age']
surveyquestions3 = df_satisfaction_xlsx[selected_columns]
# Fill missing values with mean
imputer = SimpleImputer(strategy='mean')
surveyquestions3_imputed = imputer.fit_transform(surveyquestions3)
# Set random seed for reproducibility
np.random.seed(123)
# Perform k-means clustering
kmeans = KMeans(n_clusters=7, init='k-means++', n_init=20, random_state=123)
kmeans.fit(surveyquestions3_imputed)
# Get cluster sizes
cluster_sizes = np.bincount(kmeans.labels_)
print(cluster_sizes)
#get cluster center
cluster_centers = kmeans.cluster_centers_

print(cluster_centers)

# Perform k-means clustering with 5 clusters
selected_columns = ['SeatComf', 'TimeConvenient', 'Food', 'Gate', 'InflightService', 'Entertainment', 'Support', 'OnlineBooking', 'Service', 'LegRoom', 'Baggage', 'Checkin', 'Clean', 'DelayDepart', 'DelayArrival', 'Age']
surveyquestions3 = df_satisfaction_xlsx[selected_columns]
# Fill missing values with mean
imputer = SimpleImputer(strategy='mean')
surveyquestions3_imputed = imputer.fit_transform(surveyquestions3)
# Set random seed for reproducibility
np.random.seed(123)
# Perform k-means clustering with 5 clusters
kmeans = KMeans(n_clusters=5, init='k-means++', n_init=20, random_state=123)
kmeans.fit(surveyquestions3_imputed)
# Get cluster sizes
cluster_sizes = np.bincount(kmeans.labels_)
print(cluster_sizes)

#get cluster center
cluster_centers = kmeans.cluster_centers_
print(cluster_centers)

# plot elbow curve for kmeans
selected_columns = ['SeatComf', 'TimeConvenient', 'Food', 'Gate', 'InflightService', 'Entertainment', 'Support', 'OnlineBooking', 'Service', 'LegRoom', 'Baggage', 'Checkin', 'Clean', 'DelayDepart', 'DelayArrival', 'Age']
surveyquestions3 = df_satisfaction_xlsx[selected_columns]

# Fill missing values with mean
imputer = SimpleImputer(strategy='mean')
surveyquestions3_imputed = imputer.fit_transform(surveyquestions3)
# Use elbow method to determine optimal number of clusters
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,9), metric='distortion', timings=False, locate_elbow=True)
visualizer.fit(surveyquestions3_imputed)
visualizer.show()


# pca on clustering and plotting 
# we use 3 clusters 
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=20, random_state=123)
kmeans.fit(surveyquestions3_imputed)
# Get cluster sizes
cluster_sizes = np.bincount(kmeans.labels_)
print(cluster_sizes)
#get cluster center
cluster_centers = kmeans.cluster_centers_
print(cluster_centers)
data_std = (surveyquestions3_imputed - surveyquestions3_imputed.mean()) / surveyquestions3_imputed.std()
n_components = 2
pca = PCA(n_components=2)
X_pca = pca.fit_transform(data_std)
loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i}' for i in range(1, n_components+1)], index=surveyquestions3.columns)
print(loadings)
# iterate over labels and color them
colors = ['red', 'green', 'blue']
labels = kmeans.labels_
colors = np.array(['red', 'green', 'blue'])
color_arr = colors[labels]


plt.scatter(X_pca[:,0], X_pca[:,1], c=color_arr)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()



# plot silhouette for kmeans
selected_columns = ['SeatComf', 'TimeConvenient', 'Food', 'Gate', 'InflightService', 'Entertainment', 'Support', 'OnlineBooking', 'Service', 'LegRoom', 'Baggage', 'Checkin', 'Clean', 'DelayDepart', 'DelayArrival', 'Age']
surveyquestions3 = df_satisfaction_xlsx[selected_columns]
# Fill missing values with mean
imputer = SimpleImputer(strategy='mean')
surveyquestions3_imputed = imputer.fit_transform(surveyquestions3)
# Use silhouette score to determine optimal number of clusters
model = KMeans()
visualizer = SilhouetteVisualizer(model, k=(2,9), metric='euclidean', timings=False)
visualizer.fit(surveyquestions3_imputed)
visualizer.show() 

# hierachie clustering
# Select the first 2500 rows of the DataFrame
surveyquestions3_short = surveyquestions3.iloc[:2500,:]

# Scale the selected data
scaler = StandardScaler()
stand_surveyquestions3 = scaler.fit_transform(surveyquestions3_short)

# Print the first five rows of the scaled data
print(stand_surveyquestions3[:5])
print(surveyquestions3_short.head())

# aglomarative  Bottom Up cluster analysis 
# we need to clean the data first and handle nan and infinite values
stand_surveyquestions3_clean = stand_surveyquestions3[~np.isnan(stand_surveyquestions3).any(axis=1)]
stand_surveyquestions3_clean = stand_surveyquestions3_clean[~np.isinf(stand_surveyquestions3_clean).any(axis=1)]
Z_clean = linkage(stand_surveyquestions3_clean, method='ward')
#dendrogram(Z_clean, truncate_mode='level', p=3, leaf_rotation=90, leaf_font_size=8.)

# Plot the dendrogram
plt.figure(figsize=(10, 5))
dendrogram(Z_clean, truncate_mode='level', p=3, leaf_rotation=90, leaf_font_size=8.)
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.title('Hierarchical Clustering Dendrogram')
plt.show()

# brainteaser
distances = pdist(stand_surveyquestions3_clean, metric='euclidean')
# Convert the condensed distance matrix to a squareform distance matrix
Y = squareform(distances)
# Check the shape of the condensed distance matrix
print(Y.shape)
# Use the condensed distance matrix in a clustering algorithm
Z = linkage(Y, method='ward')

dendrogram_agnes = dendrogram(Z, truncate_mode='lastp', p=50, leaf_rotation=90., leaf_font_size=8., show_contracted=True)
plt.title('Dendrogram of Airline Survey with AGNES')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()


# divison top down 
# hc_diana = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='ward', distance_threshold=0, compute_full_tree=True)
# hc_diana.fit(stand_surveyquestions3_clean)
hc_Diana_survey3 = sch.dendrogram(sch.linkage(stand_surveyquestions3_clean, method='ward'), 
                                  truncate_mode='lastp', p=50, leaf_rotation=90., leaf_font_size=8.)
plt.title('Dendrogram of Airline Survey with DIANA')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()


# checking and analyzing clustering

# Cut the dendrogram at k=5 and print the cluster sizes
sub_grp5 = cut_tree(Z, n_clusters=5).flatten()
print(pd.Series(sub_grp5).value_counts())

# Cut the dendrogram at k=6 and print the cluster sizes
sub_grp6 = cut_tree(Z, n_clusters=6).flatten()
print(pd.Series(sub_grp6).value_counts())

# Cut the dendrogram at k=4 and print the cluster sizes
sub_grp4 = cut_tree(Z, n_clusters=4).flatten()
print(pd.Series(sub_grp4).value_counts())
