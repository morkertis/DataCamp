
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage , dendrogram
from scipy.cluster.hierarchy import fcluster
from sklearn.manifold import TSNE

plt.rcParams["figure.figsize"] = (9,9)
#%%


data=pd.read_csv('data/seeds.csv').sample(n=42)
samples=data.drop(data.columns[7],axis=1).values
print(samples)

varieties=list(data[data.columns[7]].replace([1,2,3],['Kama wheat','Rosa wheat','Canadian wheat']).values)
print(varieties)




#%%



# Perform the necessary imports
from scipy.cluster.hierarchy import linkage , dendrogram
import matplotlib.pyplot as plt

# Calculate the linkage: mergings
mergings = linkage(samples,method='complete')

# Plot the dendrogram, using varieties as labels
dendrogram(mergings,
           labels=varieties,
           leaf_rotation=90,
           leaf_font_size=6,
)
plt.show()


#%%

data=pd.read_csv('data/company-stock-movements-2010-2015-incl.csv')

movements=data.drop(data.columns[0],axis=1).values
print(movements)

companies=data[data.columns[0]].values
print(companies)

#%%

'''
Note that Normalizer() is different to StandardScaler(), which you used in the previous exercise. 
While StandardScaler() standardizes features (such as the features of the fish data from the previous exercise) 
by removing the mean and scaling to unit variance, 
Normalizer() rescales each sample - here, each company's stock price - independently of the other.
'''
# Import normalize
from sklearn.preprocessing import normalize

# Normalize the movements: normalized_movements
normalized_movements = normalize(movements)

# Calculate the linkage: mergings
mergings = linkage(normalized_movements,method='complete')

# Plot the dendrogram
dendrogram(mergings,
           labels=companies,
           leaf_rotation=90,
           leaf_font_size=6,
)
plt.show()


#%%
data=pd.read_csv('data/eurovision-2016.csv').fillna(0)

data['point']=data['Jury Points']+data['Televote Points']

data=data[['From country','To country','point']]

pv=data.pivot(index='From country', columns='To country', values='point')

print(pv)

country_names=pv.index.values
print(country_names)

samples=pv.fillna(0).values
print(samples)


#%%


# Perform the necessary imports
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

# Calculate the linkage: mergings
mergings = linkage(samples,method='single')

# Plot the dendrogram
dendrogram(mergings,labels=country_names,leaf_rotation=90, leaf_font_size=6)
plt.show()


#%%
#********* use random sample *********

data=pd.read_csv('data/seeds.csv').sample(n=42)
samples=data.drop(data.columns[7],axis=1).values
print(samples)

varieties=list(data[data.columns[7]].replace([1,2,3],['Kama wheat','Rosa wheat','Canadian wheat']).values)
variety_numbers=list(data[data.columns[7]].values)
print(varieties)
print(variety_numbers)


#%%

mergings = linkage(samples,method='complete')

# Perform the necessary imports
import pandas as pd
from scipy.cluster.hierarchy import fcluster

# Use fcluster to extract labels: labels
labels = fcluster(mergings,6,criterion='distance')

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Create crosstab: ct
ct = pd.crosstab(df['labels'],df['varieties'])

# Display ct
print(ct)

#%%


# Import TSNE
from sklearn.manifold import TSNE

# Create a TSNE instance: model
model = TSNE(learning_rate=200)

# Apply fit_transform to samples: tsne_features
tsne_features = model.fit_transform(samples)

# Select the 0th feature: xs
xs = tsne_features[:,0]

# Select the 1st feature: ys
ys = tsne_features[:,1]

# Scatter plot, coloring by variety_numbers
plt.scatter(xs,ys,c=variety_numbers)
plt.show()


#%%


# Import TSNE
from sklearn.manifold import TSNE

# Create a TSNE instance: model
model = TSNE(learning_rate=50)

# Apply fit_transform to normalized_movements: tsne_features
tsne_features = model.fit_transform(normalized_movements)

# Select the 0th feature: xs
xs = tsne_features[:,0]

# Select the 1th feature: ys
ys = tsne_features[:,1]

# Scatter plot
plt.scatter(xs,ys,alpha=0.5)

# Annotate the points
for x, y, company in zip(xs, ys, companies):
    plt.annotate(company, (x, y), fontsize=5, alpha=0.75)

plt.show()


