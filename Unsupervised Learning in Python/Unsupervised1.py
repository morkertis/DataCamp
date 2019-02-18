
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

#%%


points=pd.read_csv('data/points.csv').values
new_points=pd.read_csv('data/new_points.csv').values


#%%


xs=points[:,0]
ys=points[:,1]
plt.scatter(xs,ys)
plt.show()

#%%


# Import KMeans
from sklearn.cluster import KMeans

# Create a KMeans instance with 3 clusters: model
model = KMeans(n_clusters=3)

# Fit model to points
model.fit(points)

# Determine the cluster labels of new_points: labels
labels = model.predict(new_points)

# Print cluster labels of new_points
print(labels)


#%%


# Import pyplot
import matplotlib.pyplot as plt

# Assign the columns of new_points: xs and ys
xs = new_points[:,0]
ys = new_points[:,1]

# Make a scatter plot of xs and ys, using labels to define the colors
plt.scatter(xs,ys,c=labels,alpha=0.5)

# Assign the cluster centers: centroids
centroids = model.cluster_centers_

# Assign the columns of centroids: centroids_x, centroids_y
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]

# Make a scatter plot of centroids_x and centroids_y
plt.scatter(centroids_x,centroids_y,marker='D',s=50,color='green')
plt.show()


#%%

samples=pd.read_csv('data/seeds.csv',header=None,usecols = np.arange(0,7,1)).values
varieties=pd.read_csv('data/seeds.csv',header=None,usecols = [7])

print(samples.shape)
print(samples)

varieties=list(varieties.replace([1,2,3],['Kama wheat','Rosa wheat','Canadian wheat']).values.reshape(-1))

print(type(varieties))
print(varieties)

#%%

ks = range(1, 6)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model=KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(samples)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

#%%

#my function, find the best cluster by inertias

def best_clusters_by_inertias(ks,inertias,trash=1):
    slopes=[]
    for k in range(len(ks)-1):
       slopes.append(slope(k,inertias[k],k+1,inertias[k+1])) 
    
    for i in range(len(slopes)-1):
        if slopes[i]/slopes[i+1]>=trash:
            continue
        else:
            return ks[i]
    return ks[len(ks)-2]
    
def slope(x1, y1, x2, y2):
    m = (y2-y1)/(x2-x1)
    return m

#%%
best_clusters_by_inertias(ks,inertias,trash=2)


#%%

# Create a KMeans model with 3 clusters: model
model = KMeans(n_clusters=3)

# Use fit_predict to fit model and obtain cluster labels: labels
labels = model.fit_predict(samples)

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Create crosstab: ct
ct = pd.crosstab(df['labels'],df['varieties'])

# Display ct
print(ct)


#%%

data=pd.read_csv('data/fish.csv')
samples=data.drop(data.columns[0],axis=1).values
print(samples)

species=list(data[data.columns[0]].values)
print(species)

#%%

# Perform the necessary imports
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Create scaler: scaler
scaler = StandardScaler()

# Create KMeans instance: kmeans
kmeans = KMeans(n_clusters=4)

# Create pipeline: pipeline
pipeline = make_pipeline(scaler,kmeans)


#%%
# Import pandas
import pandas as pd

# Fit the pipeline to samples
pipeline.fit(samples)

# Calculate the cluster labels: labels
labels = pipeline.predict(samples)

# Create a DataFrame with labels and species as columns: df
df = pd.DataFrame({'labels':labels,'species':species})

# Create crosstab: ct
ct = pd.crosstab(df['labels'],df['species'])

# Display ct
print(ct)


#%%

data=pd.read_csv('data/company-stock-movements-2010-2015-incl.csv')

movements=data.drop(data.columns[0],axis=1).values
print(movements)

companies=data[data.columns[0]].values
print(companies)


#%%

# Import Normalizer
from sklearn.preprocessing import Normalizer

# Create a normalizer: normalizer
normalizer = Normalizer()

# Create a KMeans model with 10 clusters: kmeans
kmeans = KMeans(n_clusters=10)

# Make a pipeline chaining normalizer and kmeans: pipeline
pipeline = make_pipeline(normalizer,kmeans)

# Fit pipeline to the daily price movements
pipeline.fit(movements)


#%%

# Import pandas
import pandas as pd

# Predict the cluster labels: labels
labels = pipeline.predict(movements)

# Create a DataFrame aligning labels and companies: df
df = pd.DataFrame({'labels': labels, 'companies': companies})

# Display df sorted by cluster label
print(df.sort_values('labels'))


