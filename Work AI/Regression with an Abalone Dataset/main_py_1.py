# %% [markdown]
# # Regression with an Abalone Dataset
# Author: Morariu Tudor

# %% [markdown]
# ## Citirea si Prelucrarea Datelor

# %%
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('dark_background')

# %%
train_data = pd.read_csv("train.csv");
test_data = pd.read_csv("test.csv");

train_data.drop(columns=['id'], inplace=True);
test_data.drop(columns=['id'], inplace=True);


# %% [markdown]
# *   **Obs 1**: Nu exista valori lipsa.
# *   **Obs 2**: Coloana "id" este inutile pentry model. :/
# *   **Obs 3**: Datele pot fi normalizate.
# 
# 

# %% [markdown]
# ### Transformarea Dateor in numere

# %%
sex_map = {
    "I": 1,
    "F": 2,
    "M": 3
}

train_data["Sex"] = train_data["Sex"].apply(lambda x: sex_map[x]);
test_data["Sex"] = test_data["Sex"].apply(lambda x: sex_map[x]);


# %% [markdown]
# ### Normalizarea datelor

# %%
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler();

train_data = pd.DataFrame(scaler.fit_transform(train_data));
test_data = pd.DataFrame(scaler.fit_transform(test_data));

train_data["Sex"] = train_data[0];
train_data["Length"] = train_data[1];
train_data["Diameter"] = train_data[2];
train_data["Height"] = train_data[3];
train_data["Whole weight"] = train_data[4];
train_data["Whole weight.1"] = train_data[5];
train_data["Whole weight.2"] = train_data[6];
train_data["Shell weight"] = train_data[7];
train_data["Rings"] = train_data[8];
train_data.drop(columns=[0, 1, 2, 3, 4, 5, 6, 7, 8], inplace=True);

test_data["Sex"] = test_data[0];
test_data["Length"] = test_data[1];
test_data["Diameter"] = test_data[2];
test_data["Height"] = test_data[3];
test_data["Whole weight"] = test_data[4];
test_data["Whole weight.1"] = test_data[5];
test_data["Whole weight.2"] = test_data[6];
test_data["Shell weight"] = test_data[7];
test_data.drop(columns=[0, 1, 2, 3, 4, 5, 6, 7], inplace=True);



# %% [markdown]
# ### Plotarea Datelor si a Corelatiilor dintre ele

# %%
train_data.hist(figsize=(20, 20));

# %%
corr = train_data.corr()
corr.style.background_gradient(cmap='coolwarm')


# %% [markdown]
# *   **Obs**: In general toate datele sunt corelate intre ele, cu exceptia coloanei "Rings"

# %% [markdown]
# ### Plot folosind PCA

# %%
from sklearn.decomposition import PCA

ax = plt.figure().add_subplot(projection='3d')

pca = PCA(n_components=3);

points = pca.fit_transform(train_data)
pointsx = [];
pointsy = [];
pointsz = [];

for x, y, z in points:
    pointsx.append(x);
    pointsy.append(y);
    pointsz.append(z);

ax.plot(pointsx, pointsy, pointsz);

plt.show();
# %% [markdown]
# ### Train-Test Split

# %%
from sklearn.model_selection import train_test_split

train_f, test_f, train_l, test_l = train_test_split(train_data.drop(columns=["Rings"]), train_data["Rings"], test_size=0.3);

# %% [markdown]
# ## Alegerea si Antrenarea Modelului

# %% [markdown]
# ### Modelul

# %% [markdown]
# Pentru model am putea folosi Linear Regression.

# %%
from sklearn.linear_model import LinearRegression

model = LinearRegression()


