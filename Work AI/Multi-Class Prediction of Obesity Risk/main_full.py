# %% [markdown]
# # <center> Multi-Class Prediction of Obesity Risk </center>
# <center> Author: Morariu Tudor </center>

# %% [markdown]
# ## Citirea si Procesarea Datelor

# %%
import pandas as pd
pd.set_option('display.max_columns', None)

data = pd.read_csv("train.csv");


# %% [markdown]
# Printez pentru fiecare coloana ce elemente unice am.

# %%
for c in data.columns:
    print(c, data[c].unique());

# %%
Gender_map = {
    "Male": 1,
    "Female": 0
}

yes_no_map = {
    "yes": 1,
    "no" : 0
}

CAEC_map = {
    'no': 1,
    'Sometimes': 2,
    'Frequently': 3,
    'Always': 4
}

CALC_map = {
    'no': 1,
    'Sometimes': 2,
    'Frequently': 3
}

MTRANS_map = {
    'Automobile': 1,
    'Public_Transportation': 2,
    'Motorbike': 3,
    'Bike': 4,
    'Walking': 5
}

NObeyesdad_map = {
    'Insufficient_Weight': 1,
    'Normal_Weight': 2,
    'Overweight_Level_I': 3,
    'Overweight_Level_II': 4,
    'Obesity_Type_I': 5,
    'Obesity_Type_II': 6,
    'Obesity_Type_III': 7
}


# %% [markdown]
#  Pentru fircare coloana trebuie sa transform din *object* → *int*. 
# 
# Functia *LabelEncoder* din *sklearn* nu este buna pentru acesta problema pentru ca va codifica **yes**: $0$ si **no**: $1$ pe o coloana si pe alta **yes**: $1$ si **no**: $0$ in functie de prima aparitie. 
# 
# Am definit cateva map-uri cu valorile dataset-ului in functie de ce "severitate" au. Spre exemplu **Motorbike**: $3$ si **Walking**: $5$ pentru ca un om care merge are exercita mai mult efort decat un om care merge pe motocicleta.

# %%
data["Gender"] = data["Gender"].apply(lambda x: Gender_map[x]);
data["family_history_with_overweight"] = data["family_history_with_overweight"].apply(lambda x: yes_no_map[x]);
data["FAVC"] = data["FAVC"].apply(lambda x: yes_no_map[x]);
data["CAEC"] = data["CAEC"].apply(lambda x: CAEC_map[x]);
data["SMOKE"] = data["SMOKE"].apply(lambda x: yes_no_map[x]);
data["SCC"] = data["SCC"].apply(lambda x: yes_no_map[x]);
data["CALC"] = data["CALC"].apply(lambda x: CALC_map[x]);
data["MTRANS"] = data["MTRANS"].apply(lambda x: MTRANS_map[x]);
data["NObeyesdad"] = data["NObeyesdad"].apply(lambda x: NObeyesdad_map[x]);


# %% [markdown]
# ### Pentru a intelege mai bine datasetul voi plota matricea de corelatie

# %%
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('dark_background')


f, ax = plt.subplots(figsize=(10, 10))
corr = data.corr()
sns.heatmap(corr, cmap=sns.diverging_palette(220, 10, as_cmap=True), vmin=-1.0, vmax=1.0, square=True, ax=ax)

# %% [markdown]
# *   **Obs**: Raspunsul este corelat puternic cu **Weight** (evident).

# %% [markdown]
# ### Plotare $2D$ folosind PCA

# %%
from sklearn.decomposition import PCA

f, ax = plt.subplots(figsize=(10, 10))

pca2d = PCA(n_components=2);
points2d = pca2d.fit_transform(data);


colors2d = ['#003bff', '#05ff00', '#ffb4b4', '#ff8888', '#ff6565', '#fe4545', '#ff0000']
#         Underweight,  Normal,  Overweight1,  Overweight2,  Obese1,  Obese2,  Obese3.

list_dateframe_index = data.index;
iterator = 0;

points2d_under_x = [];
points2d_under_y = [];

points2d_norml_x = [];
points2d_norml_y = [];

points2d_over1_x = [];
points2d_over1_y = [];

points2d_over2_x = [];
points2d_over2_y = [];

points2d_obst1_x = [];
points2d_obst1_y = [];

points2d_obst2_x = [];
points2d_obst2_y = [];

points2d_obst3_x = [];
points2d_obst3_y = [];

for x, y in points2d:
    #print(data.loc[list_dateframe_index[iterator], "NObeyesdad"]);
    if(data.loc[list_dateframe_index[iterator], "NObeyesdad"] == 1):
        points2d_under_x.append(x);
        points2d_under_y.append(y);
    
    if(data.loc[list_dateframe_index[iterator], "NObeyesdad"] == 2):
        points2d_norml_x.append(x);
        points2d_norml_y.append(y);
    
    if(data.loc[list_dateframe_index[iterator], "NObeyesdad"] == 3):
        points2d_over1_x.append(x);
        points2d_over1_y.append(y);
    
    if(data.loc[list_dateframe_index[iterator], "NObeyesdad"] == 4):
        points2d_over2_x.append(x);
        points2d_over2_y.append(y);
    
    if(data.loc[list_dateframe_index[iterator], "NObeyesdad"] == 5):
        points2d_obst1_x.append(x);
        points2d_obst1_y.append(y);
    
    if(data.loc[list_dateframe_index[iterator], "NObeyesdad"] == 6):
        points2d_obst2_x.append(x);
        points2d_obst2_y.append(y);
    
    if(data.loc[list_dateframe_index[iterator], "NObeyesdad"] == 7):
        points2d_obst3_x.append(x);
        points2d_obst3_y.append(y);
    iterator += 1;

points2d_size = 0.5;

ax.scatter(points2d_under_x, points2d_under_y, c=colors2d[0], s=points2d_size);
ax.scatter(points2d_norml_x, points2d_norml_y, c=colors2d[1], s=points2d_size);
ax.scatter(points2d_over1_x, points2d_over1_y, c=colors2d[2], s=points2d_size);
ax.scatter(points2d_over2_x, points2d_over2_y, c=colors2d[3], s=points2d_size);
ax.scatter(points2d_obst1_x, points2d_obst1_y, c=colors2d[4], s=points2d_size);
ax.scatter(points2d_obst2_x, points2d_obst2_y, c=colors2d[5], s=points2d_size);
ax.scatter(points2d_obst3_x, points2d_obst3_y, c=colors2d[6], s=points2d_size);

plt.show();


# %% [markdown]
# ### Plot $3D$ cu PCA

# %%


# %%

fig = plt.figure(figsize=(10, 10))

pca3d = PCA(n_components=3);
points2d = pca3d.fit_transform(data);


colors2d = ['#003bff', '#05ff00', '#ffb4b4', '#ff8888', '#ff6565', '#fe4545', '#ff0000']
#         Underweight,  Normal,  Overweight1,  Overweight2,  Obese1,  Obese2,  Obese3.

list_dateframe_index = data.index;
iterator = 0;

points3d_under_x = [];
points3d_under_y = [];
points3d_under_z = [];


points3d_norml_x = [];
points3d_norml_y = [];
points3d_norml_z = [];

points3d_over1_x = [];
points3d_over1_y = [];
points3d_over1_z = [];

points3d_over2_x = [];
points3d_over2_y = [];
points3d_over2_z = [];

points3d_obst1_x = [];
points3d_obst1_y = [];
points3d_obst1_z = [];

points3d_obst2_x = [];
points3d_obst2_y = [];
points3d_obst2_z = [];

points3d_obst3_x = [];
points3d_obst3_y = [];
points3d_obst3_z = [];

for x, y, z in points2d:
    #print(data.loc[list_dateframe_index[iterator], "NObeyesdad"]);
    if(data.loc[list_dateframe_index[iterator], "NObeyesdad"] == 1):
        points3d_under_x.append(x);
        points3d_under_y.append(y);
        points3d_under_z.append(z);
    
    if(data.loc[list_dateframe_index[iterator], "NObeyesdad"] == 2):
        points3d_norml_x.append(x);
        points3d_norml_y.append(y);
        points3d_norml_z.append(z);
    
    if(data.loc[list_dateframe_index[iterator], "NObeyesdad"] == 3):
        points3d_over1_x.append(x);
        points3d_over1_y.append(y);
        points3d_over1_z.append(z);
    
    if(data.loc[list_dateframe_index[iterator], "NObeyesdad"] == 4):
        points3d_over2_x.append(x);
        points3d_over2_y.append(y);
        points3d_over2_z.append(z);
    
    if(data.loc[list_dateframe_index[iterator], "NObeyesdad"] == 5):
        points3d_obst1_x.append(x);
        points3d_obst1_y.append(y);
        points3d_obst1_z.append(z);
    
    if(data.loc[list_dateframe_index[iterator], "NObeyesdad"] == 6):
        points3d_obst2_x.append(x);
        points3d_obst2_y.append(y);
        points3d_obst2_z.append(z);
    
    if(data.loc[list_dateframe_index[iterator], "NObeyesdad"] == 7):
        points3d_obst3_x.append(x);
        points3d_obst3_y.append(y);
        points3d_obst3_z.append(z);
    
    iterator += 1;

points3d_size = 0.5;

init_views = [(0, 0)];

for i in range(4):
    ax = fig.add_subplot(2, 2, i + 1, projection='3d')
    #ax.init_view(init_views[0]);
    ax.scatter(points3d_under_x, points3d_under_y, points3d_under_z, c=colors2d[0], s=points3d_size);
    ax.scatter(points3d_norml_x, points3d_norml_y, points3d_norml_z, c=colors2d[1], s=points3d_size);
    ax.scatter(points3d_over1_x, points3d_over1_y, points3d_over1_z, c=colors2d[2], s=points3d_size);
    ax.scatter(points3d_over2_x, points3d_over2_y, points3d_over2_z, c=colors2d[3], s=points3d_size);
    ax.scatter(points3d_obst1_x, points3d_obst1_y, points3d_obst1_z, c=colors2d[4], s=points3d_size);
    ax.scatter(points3d_obst2_x, points3d_obst2_y, points3d_obst2_z, c=colors2d[5], s=points3d_size);
    ax.scatter(points3d_obst3_x, points3d_obst3_y, points3d_obst3_z, c=colors2d[6], s=points3d_size);




plt.show();


# %% [markdown]
# ## Alegerea si Procesarea Modelului


