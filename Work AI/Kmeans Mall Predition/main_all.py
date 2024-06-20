# %% [markdown]
# # K Means Mall spending prediciton
# Morariu Tudor

# %% [markdown]
# ### Importam librariile necesare.

# %%
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# %%
from time import gmtime, strftime

def watermark():
  return "Tudor Morariu, generated at: " + strftime("%H:%M:%S", gmtime());

# %%
# Definim dark-mode pentru matplotlib pentru ca e cool
plt.style.use('dark_background')

# %% [markdown]
# ### Citim dataset-ul

# %%
data = pd.read_csv("Mall_Customers.csv");

# %% [markdown]
# *  Coloana "Gender" este un string. Trebuie sa il transformam in float.
# *  Coloana "CustomerID" este inutila.
# *  Folosim Label Encoder din sklearn pentru a encoda $Male$ in $1$, $Female$ in $0$.
# 

# %%
le = LabelEncoder()

data.drop(columns=['CustomerID'], inplace=True);
data['Gender'] = le.fit_transform(data['Gender']);


# %% [markdown]
# ## Plotam datele.

# %% [markdown]
# ### Histograme si distributie

# %%
# Facem o histograma pentru fiecare feture pentru a vedea distributia datelor.
data.hist(figsize=(10, 10));

# %% [markdown]
# ### PCA

# %% [markdown]
# Vom face un plot $2D$ si $3D$ al feature-urilor folosind $PCA$ pentru a vizualiza clusterele.

# %%
from sklearn.decomposition import PCA

pca3d = PCA(n_components=3);
pca2d = PCA(n_components=2);

pca3d.fit(data);
pca2d.fit(data);

points3d = pca3d.fit_transform(data);
points2d = pca2d.fit_transform(data);

# %%
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

pointsx = [];
pointsy = [];
pointsz = [];

for x, y, z in points3d:
    pointsx.append(x);
    pointsy.append(y);
    pointsz.append(z);


ax.scatter(pointsx, pointsy, pointsz, marker='o')

# %%
pointsx = [];
pointsy = [];

for x, y in points2d:
    pointsx.append(x);
    pointsy.append(y);

plt.plot(pointsx, pointsz, 'ro')

# %% [markdown]
# ### Corelatie

# %%
# Plotam matricea de coreleatie.
plt.matshow(data.corr())
plt.show()

# %% [markdown]
# *   **Obs**: Din matricea de corelatie observam ca "Spending Score" depinde de "Age"

# %%
data.describe()


# %%
data.info()

# %% [markdown]
# Toate valorile sunt $int$ deci putem normaliza

# %% [markdown]
# ## Pregatirea Modelului

# %% [markdown]
# ### Normalizare

# %%
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

data = pd.DataFrame(scaler.fit_transform(data))

# %%
data.describe()

# %% [markdown]
# ### K Means

# %%
from sklearn.cluster import KMeans

error = []

for i in range(1, 7):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(data)

    error.append(kmeans.inertia_)

plt.plot(range(1, 7), error)
plt.title('The Elbow Method')
plt.xlabel('numar de centroide')
plt.ylabel('eroarea')
plt.text(0, 0, watermark(), fontsize=10, color="white", ha="right", va="top", alpha=0.3, rotation=0)

plt.show()

# %% [markdown]
# Observam ca numarul optim de clutere este 5.

# %% [markdown]
# ### Performanta Modelului

# %%
from sklearn.metrics import silhouette_score

# %%
silhouette_scores = [];

for centr in range(2, 100):
    kmeans = KMeans(n_clusters=centr, init='k-means++', max_iter=300, n_init=10, random_state=1)
    y_kmeans = kmeans.fit_predict(data)
    silhouette_avg = silhouette_score(data, y_kmeans)
    silhouette_scores.append(silhouette_avg);

plt.plot(range(2, 100), silhouette_scores)
plt.title('Silhouette score plot')
plt.xlabel('numar de centroide')
plt.ylabel('Silhouette score')
plt.text(0, 0.25, watermark(), fontsize=10, color="white", ha="right", va="top", alpha=0.3, rotation=0)


# %% [markdown]
# *   **Obs**: Numarul optim de centroide este $5$, pentru a evita overfitting-ul

# %% [markdown]
# ## Plotare 3D cu centroide

# %% [markdown]
# ### Obtinerea Centroizilor

# %%
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=1)
y_kmeans = kmeans.fit_predict(data)

centroids = kmeans.cluster_centers_

print(centroids)

# %% [markdown]
# ### PCA pentru datele normalizate si centroizi

# %%
pca3d = PCA(n_components=3).fit(data);

points3d = pca3d.fit_transform(data);
points3d_centroids = pca3d.fit_transform(centroids);

# %%
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

pointsx = [];
pointsy = [];
pointsz = [];

for x, y, z in points3d:
    pointsx.append(x);
    pointsy.append(y);
    pointsz.append(z);


ax.scatter(pointsx, pointsy, pointsz, marker='o')

for x_cen, y_cen, z_cen in points3d_centroids:
    center = (x_cen, y_cen, z_cen)
    radius = 0.2
    phi = np.linspace(0, np.pi, 100)
    theta = np.linspace(0, 2 * np.pi, 100)

    # Create a mesh grid for the sphere
    phi, theta = np.meshgrid(phi, theta)
    x = center[0] + radius * np.sin(phi) * np.cos(theta)
    y = center[1] + radius * np.sin(phi) * np.sin(theta)
    z = center[2] + radius * np.cos(phi)
    ax.plot_wireframe(x, y, z, color="r")

plt.show();
