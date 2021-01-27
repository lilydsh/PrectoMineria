
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# %matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')


#Insertamos el arcchivo y lo convertimos a un dataframe
datos = pd.read_csv('interrupcion-legal-del-embarazo.csv')
df= pd.DataFrame(datos)

#checamos los primeros datos y como se ven las estadisticas
df.head()

df.describe()


# los datos vacios los llenamos con 0 y los convetimos a int para poder trabajar más fácil
df = df.fillna(0)
df[['edad','año','fsexual','nhijos','npartos','naborto']].astype('int')

print(df.groupby('edad').size())

df.drop(['edad'],1).hist()
plt.show()

# Graficamos la edad junto con la frecuencia sexual, el número de hijos y el numero de abortos para ver si hay alguna relacion
sb.pairplot(df.dropna(), hue='edad',size=4,vars=["fsexual","nhijos","naborto"],kind='scatter')

X = np.array(df[["fsexual","nhijos","naborto"]])
y = np.array(df['edad'])
X.shape

# aplicamos k-means
Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans
score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]
score
plt.plot(Nc,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()
Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans
score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]
score
plt.plot(Nc,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()


#Definimos el número de clusters
kmeans = KMeans(n_clusters=5).fit(X)
centroids = kmeans.cluster_centers_
print(centroids)

#Predecimos los clusters
labels = kmeans.predict(X)
# Sacamos los centros de los clusters
C = kmeans.cluster_centers_
colores=['red','green','blue','cyan','yellow']
asignar=[]
for row in labels:
    asignar.append(colores[row])

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=asignar,s=60)
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c=colores, s=1000)

# De los valores ya los graficamos
f1 = df['fsexual'].values
f2 = df['naborto'].values

plt.scatter(f1, f2, c=asignar, s=70)
plt.scatter(C[:, 0], C[:, 1], marker='*', c=colores, s=1000)
plt.show()


f1 = df['fsexual'].values
f2 = df['naborto'].values

plt.scatter(f1, f2, c=asignar, s=70)
plt.scatter(C[:, 0], C[:, 1], marker='*', c=colores, s=1000)
plt.show()


f1 = df['edad'].values
f2 = df['naborto'].values

plt.scatter(f1, f2, c=asignar, s=70)
plt.scatter(C[:, 0], C[:, 2], marker='*', c=colores, s=1000)
plt.show()

f1 = df['nhijos'].values
f2 = df['naborto'].values

plt.scatter(f1, f2, c=asignar, s=70)
plt.scatter(C[:, 1], C[:, 2], marker='*', c=colores, s=1000)
plt.show()
