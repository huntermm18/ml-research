import matplotlib.pyplot as plt

plt.ion()
# import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm

df = pd.read_csv("./results_simple_multipass_text-davinci-003.csv")

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
fig.write_html("./result_viz/index.html")
