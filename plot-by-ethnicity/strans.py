
# pip install --upgrade numpy
# pip install --upgrade numba
# pip install matplotlib seaborn umap-learn sentence_transformers

import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
from scipy import linalg as LA

plt.ion()

# ======================================================================================================

def soPCA(data, dims_rescaled_data=2):
    """
    returns: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """
    m, n = data.shape
    # mean center the data
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = np.cov(data, rowvar=False)
    # caplculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric,
    # the performance gain is substantial
    evals, evecs = LA.eigh( R )
    # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dims_rescaled_data]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return np.dot(evecs.T, data.T).T, evals, evecs

# ======================================================================================================

df = pd.read_csv("./results_simple_multipass_text-davinci-003.csv")

sentences = []
for pid in tqdm(range(len(df))):
    person = df.iloc[pid]
    sentences.append( person['response'] )

# ======================================================================================================

model = SentenceTransformer('sentence-t5-xxl')
sentence_embeddings = model.encode( sentences )
np.save("./nsyr_embs.npy",sentence_embeddings)

# ======================================================================================================

sse_pca = soPCA( sentence_embeddings, 50 )

umap = UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.001,
    spread=1.0,
    metric='cosine',
    init='random',
    random_state=0
)
projections = umap.fit_transform( sse_pca[0] )

# start clustering here (dbscan,  agl clustering)

np.save( "./simple_multipass_pca50.npy", sse_pca[0] )
np.save( "./simple_multipass_projections.npy", projections )

fig = px.scatter(projections, x=0, y=1,)
fig.show()

