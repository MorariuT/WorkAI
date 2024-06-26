# %% [markdown]
# # Red Wine Quality
# Author: Morariu Tudor

# %% [markdown]
# ## Citirea si procesarea Datelor

# %%
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("winequality-red.csv");


# %% [markdown]
# ## Plotarea Datelor

# %%
data.hist(figsize=(10, 10));

# %% [markdown]
# ## Plot cu PCA

# %%
from sklearn.decomposition import PCA

ax = plt.figure().add_subplot(projection='3d')

pca = PCA(n_components=3);

points = pca.fit_transform(data.drop(columns=["quality"]));

pointsx = [];
pointsy = [];
pointsz = [];

for x, y, z in points:
    pointsx.append(x);
    pointsy.append(y);
    pointsz.append(z);

ax.scatter(pointsx, pointsy, pointsz, s=1);
plt.show();


# %% [markdown]
# ## Regresie

# %%
from xgboost import XGBRegressor



