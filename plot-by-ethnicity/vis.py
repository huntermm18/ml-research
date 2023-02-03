import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.ion()

OUTPUT_FILE_NAMES  = ['plotly-graph.html', 'output-graphs.pdf', 'plotly-graph-clusters.html']

df = pd.read_csv("./results_simple_multipass_text-davinci-003.csv")

# ------------------- get sentiment data -------------------

import matplotlib.backends.backend_pdf
from textblob import TextBlob
pdf = matplotlib.backends.backend_pdf.PdfPages(f"./result_viz/{OUTPUT_FILE_NAMES[1]}")

# Function to calculate sentiment polarity
def sentiment_polarity(text, includ_neutral=False):
    sentiment = TextBlob(text).sentiment.polarity
    if sentiment < 0:
        return "Negative"
    elif sentiment == 0 and includ_neutral:
        return "Neutral"
    else:
        return "Positive"

stories = {}
sentiment_means = {}
ethnicities = df['ethnicity'].unique()
for e in ethnicities:
    stories[e] = None
for e in ethnicities:
    stories[e] = df['response'].loc[(df['ethnicity'] == e)]
for e in ethnicities:
    s = []
    for response in stories[e]:
        s.append(TextBlob(response).sentiment.polarity)
    sentiment_means[e] = np.mean(s)

# matplot lib bar chart
import matplotlib.pyplot as plt
data = sentiment_means
ind = np.arange(len(data))
fig = plt.figure()
plt.bar(ind, list(data.values()))
plt.xticks(ind, list(data.keys()))
plt.show()
pdf.savefig(fig)

# pie charts
for ethnicity in stories.keys():
  sentiment_list = [sentiment_polarity(story) for story in stories[ethnicity]]

  sentiment_keys = ['Positive', 'Negative']
  if len(sentiment_list) == 0:
    print('sentiment list empty: ', ethnicity)
    continue
  values = [sentiment_list.count('Positive') / len(sentiment_list), sentiment_list.count('Negative') / len(sentiment_list)]

  # Plotting the results as a pie chart
  fig = plt.figure()
  plt.pie(values, labels=sentiment_keys, startangle=90, counterclock=False,
          autopct='%1.1f%%', shadow=True)
  plt.axis('equal')
  plt.title(f'Sentiment Analysis Results: {ethnicity}')
  plt.show()
  pdf.savefig(fig)

# save figures to pdf
pdf.close()
# ------------------- End get sentiment data -------------------

sentences = []
for pid in tqdm(range(len(df))):
    person = df.iloc[pid]
    sentences.append(person['response'])

rid = 0
rids = {}
rrids = {}
rtexts = []
rcodes = []

for pid in tqdm(range(len(df))):
    person = df.iloc[pid]
    ethnicity = person['ethnicity']

    if type(ethnicity) != str:
        ethnicity = 'Other'
    ethnicity = ethnicity.strip().lower()
    rtexts.append(ethnicity)
    # print(ethnicity)
    if not ethnicity in rids:
        rids[ethnicity] = rid
        rid += 1

    rcodes.append(rids[ethnicity])

rcodes = np.array(rcodes)

for k, v in rids.items():
    rrids[v] = k

projections = np.load("./simple_multipass_projections.npy")

plt.clf()

legend_entries = []
for r in range(rid):
    inds = rcodes == r
    legend_entries.append(rrids[r])
    plt.scatter(projections[inds, 0], projections[inds, 1], alpha=0.5)
plt.legend(legend_entries)
# ---------------------- clustering -------------------------------------------------
from sklearn.cluster import KMeans

def cluster_data(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(data)
    return kmeans.labels_

cluster_labels = cluster_data(projections, 5)
# ---------------------- clustering -------------------------------------------------
# ---------------------- plotly -------------------------------------------------
# import plotly
import plotly.express as px


# add linebreaks
def split_string(string, parts=4):
    n = len(string)
    return [string[i*n//parts:(i+1)*n//parts] for i in range(parts)]


s1, s2, s3, s4 = [], [], [], []
for s in sentences:
    x = split_string(s, 4)
    s1.append(x[0])
    s2.append(x[1])
    s3.append(x[2])
    s4.append(x[3])

fig = px.scatter(
    projections,
    x=0, y=1,
    #    color=[str(x) for x in rcodes],
    color=rtexts,
    color_discrete_sequence=px.colors.qualitative.Prism,
    # hover_name=np.array(sentences), # was long, so I broke into 4 parts
    hover_data=[s1, s2, s3, s4]
)
# fig.show()
fig.write_html(f"./result_viz/{OUTPUT_FILE_NAMES[0]}")

# color by cluster
fig = px.scatter(
    projections,
    x=0, y=1,
    color=cluster_labels,
    color_discrete_sequence=px.colors.qualitative.Prism,
    hover_name=rtexts,
    hover_data=[s1, s2, s3, s4]
)
fig.write_html(f"./result_viz/{OUTPUT_FILE_NAMES[2]}")
