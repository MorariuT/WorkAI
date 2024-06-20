# %%
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import cv2 as cv
import pandas as pd
import tensorflow as tf
import numpy as np
import keras
import nltk
from matplotlib import pyplot as plt
from textblob import TextBlob 
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from datetime import date
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import requests
import matplotlib.pyplot as plt

# %%
polarity_mapping = {"Extremely Negative": -2,
                    "Negative": -1,
                    "Neutral": 0,
                    "Positive": 1,
                    "Extremely Positive": 2};

inv_polarity_mapping = {-2: "Extremely Negative",
                        -1: "Negative",
                        0: "Neutral",
                        1: "Positive",
                        2: "Extremely Positive"}

data_sentiments = []

sentiments = ["Extremely Negative",  "Negative", "Neutral", "Positive", "Extremely Positive"];

hashtags = [];

count = 0;
sentiment_search = "";

train_features = [];
train_labels = [];

negative_data = [];
neutral_data = [];
positive_data = [];

analyzer = SentimentIntensityAnalyzer()

# %%
def remove_ats_and_links(row):
    text = row["OriginalTweet"];
    text = text.split();

    new_text = "";

    for word in text:
        
        all_ascii = True

        for letter in word:
            if(letter.isascii() == False):
                all_ascii = False;
                break;

        if(word[0] == '@' or ("http" in word) or not all_ascii): continue;

        new_text += word + " ";
    row["EditiedTweet"] = new_text;
    return row;

def process_hashtags(row):
    text = row["EditiedTweet"].split();
    new_text = ""

    for word in text:
        if(word[0] == '#'):
            hashtags.append(word[1:]);
            continue
        new_text += word + " ";
    row["EditiedTweet"] = new_text;
    return row;

def make_features_and_labels(row):
    text = row["EditiedTweet"];
    sentiment = analyzer.polarity_scores(text);
    try:
        train_labels.append(polarity_mapping[row["Sentiment"]]);
    
        if("Negative" in row["Sentiment"]):
            negative_data.append(sentiment['neg']);
            negative_data.append(sentiment['neu']);
            negative_data.append(sentiment['pos']);
            #negative_data.append(sentiment['compound']);
        if("Neutral" in row["Sentiment"]):
            neutral_data.append(sentiment['neg']);
            neutral_data.append(sentiment['neu']);
            neutral_data.append(sentiment['pos']);
            #neutral_data.append(sentiment['compound']);
        if("Positive" in row["Sentiment"]):
            positive_data.append(sentiment['neg']);
            positive_data.append(sentiment['neu']);
            positive_data.append(sentiment['pos']);
            #positive_data.append(sentiment['compound']);

    except KeyError:
        pass;
    train_features.append(sentiment['neg']);
    train_features.append(sentiment['neu']);
    train_features.append(sentiment['pos']);
    train_features.append(sentiment['compound']);
    
    return row


# %%

train = pd.read_csv("Pandemic_NLP_train.csv");
test = pd.read_csv("Pandemic_NLP_test_.csv");

train = train.head(1000);

# %%


# %%
# display(train);
## display(test);

# %%
train = train.apply(remove_ats_and_links, axis=1);
train = train.apply(process_hashtags, axis=1);
train = train.apply(make_features_and_labels, axis=1);

# %%
# display(train);

print(negative_data)

# %%
train_features = np.array(train_features).reshape(-1, 4);
train_labels = np.array(train_labels);

# %%
negative_data = np.array(negative_data).reshape(-1, 3);
neutral_data = np.array(neutral_data).reshape(-1, 3);
positive_data = np.array(positive_data).reshape(-1, 3);

print(negative_data)

# %%
pca = PCA(n_components=2)
pca_results_negative = pca.fit_transform(negative_data)
pca_results_neutral = pca.fit_transform(neutral_data)
pca_results_positive = pca.fit_transform(positive_data)

# %%
print(pca_results_negative);

# %%
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for x, y, z in negative_data:
    ax.scatter(x, y, z, '^', c='k');
for x, y, z in neutral_data:
    ax.scatter(x, y, z, '^', c='b');
for x, y, z in positive_data:
    ax.scatter(x, y, z, 'r^', c='r');


#ax.scatter(negative_data);
#ax.scatter(positive_data);
#ax.scatter(neutral_data);
#plt.plot(neutral_data, 'b.', 0.1);
#plt.plot(positive_data, 'r.', 0.1);
plt.show();
'''

# %%


# %%
print(train_features)
print(train_labels)

# %%
model = MLPClassifier(verbose=True, max_iter=100000, tol=0.0001, learning_rate='adaptive').fit(train_features, train_labels);

# %%
print(analyzer.polarity_scores("test"))

# %%
train_features = [];
train_labels = [];

# %%
test = test.apply(remove_ats_and_links, axis=1);
test = test.apply(process_hashtags, axis=1);
tset = test.apply(make_features_and_labels, axis=1);

# %%
train_features = np.array(train_features).reshape(-1, 4);
train_labels = np.array(train_labels);

test_features = train_features;
test_labels = train_labels;

# %%
ans = model.predict(test_features);
ans_string = [];

# %%
for i in range(len(ans)):
    ans_string.append(inv_polarity_mapping[ans[i]]);


# %%
df_ans = pd.DataFrame();

df_ans["OriginalTweet"] = test["OriginalTweet"];
df_ans["Sentiment"] = ans_string;

df_ans.to_csv("ans.csv", index=False);

# display(df_ans);

# %%



'''