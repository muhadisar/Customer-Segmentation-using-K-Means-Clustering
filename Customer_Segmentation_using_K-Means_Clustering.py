import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Load the data
data = pd.read_csv('Wholesale customers data.csv')
print(data)

# Data Preprocessing: 
# Droping columns that is not needed in my trend analysis
data_clean = data.drop(columns=['Channel', 'Region'])

# Standardization of the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_clean)



# KMeans
# After tying differnet k value for silhouette_score, 3 was the optimal k value 
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['Cluster'] = kmeans.fit_predict(data_scaled)


# Evaluating clustering with Silhouette Score
sil_score = silhouette_score(data_scaled, data['Cluster'])
print(f'Silhouette Score: {sil_score}')


#Observed centroids in DataFrame
centroids = pd.DataFrame(kmeans.cluster_centers_, columns=data_clean.columns)



#Visualizing the clusters using a 2D PCA plot
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

# Plotting the data points for each cluster
for cluster in range(optimal_k):
    plt.scatter(
        data_pca[data['Cluster'] == cluster, 0], 
        data_pca[data['Cluster'] == cluster, 1], 
        label=f"Cluster {cluster}"
    )

# Transforming the centroids to the PCA space
centroids_pca = pca.transform(centroids)  


# Plotting the centroids in the 2D PCA space
plt.scatter(
    centroids_pca[:, 0], centroids_pca[:, 1], 
    c='black', marker='X', s=200, label='Centroids'
)

# Adding labels and title
plt.title('Customer Spendings')
plt.xlabel('Prinicpal Component 1')
plt.ylabel('Principal Component 2')

# Adding a legend
plt.legend()

# Showing the plot
plt.show()




