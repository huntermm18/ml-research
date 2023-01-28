import matplotlib.pyplot as plt
plt.ion()
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm


df1 = pd.read_csv("./results_code_simple_multipass_text-davinci-003.csv")
df2 = pd.read_csv("./results_simple_multipass_text-davinci-003.csv")

sentences = []
for pid in tqdm( range(len(df2)) ):
    person = df2.iloc[pid]
    sentences.append( person['response'] )

rid = 0
rids = {}
rrids = {}
rtexts = []
rcodes = []

for pid in tqdm( range(len(df1)) ):
    person = df1.iloc[pid]
    reason_code = person['code']

    reason_code = reason_code.strip().lower()
    rtexts.append( reason_code )
    print( reason_code )
    if not reason_code in rids:
        rids[reason_code] = rid
        rid += 1

    rcodes.append( rids[reason_code] )

rcodes = np.array(rcodes)

for k,v in rids.items():
    rrids[v] = k

projections = np.load("./simple_multipass_projections.npy")

plt.clf()

legend_entries = []
for r in range( rid ):
    inds = rcodes == r
    legend_entries.append( rrids[r] )
    plt.scatter( projections[inds,0], projections[inds,1], alpha=0.5 )
plt.legend( legend_entries )

import plotly
import plotly.express as px

fig = px.scatter(
    projections,
    x=0, y=1,
#    color=[str(x) for x in rcodes],
    color=rtexts,
    color_discrete_sequence=px.colors.qualitative.Prism,
    hover_name = np.array(sentences)
)
fig.show()


fig.write_html( "./result_viz/index.html" )
