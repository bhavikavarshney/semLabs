# DBSCAN (Density Based Spatial Clustering of Applications with Noise)
# Importing libraries
import numpy as nmp
import pandas as pds
import matplotlib.pyplot as pplt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

M = pds.read_csv('sampleDataset.csv')
M = M.drop('CUST_ID', axis = 1)
M.fillna(method ='ffill', inplace = True)
print(M.head())

scalerFD = StandardScaler()
M_scaled = scalerFD.fit_transform(M)
M_normalized = normalize(M_scaled)
M_normalized = pds.DataFrame(M_normalized)
pcaFD = PCA(n_components = 2) 
M_principal = pcaFD.fit_transform(M_normalized)
M_principal = pds.DataFrame(M_principal)
M_principal.columns = ['C1', 'C2']
print(M_principal.head())


db_default = DBSCAN(eps = 0.0375, min_samples = 3).fit(M_principal)
labeling = db_default.labels_

colours = {}
colours[0] = 'g'
colours[1] = 'k'
colours[2] = 'r'
colours[-1] = 'b'
cvec = [colours[label] for label in labeling]

g = pplt.scatter(M_principal['C1'], M_principal['C2'], color ='g');
k = pplt.scatter(M_principal['C1'], M_principal['C2'], color ='k');
r = pplt.scatter(M_principal['C1'], M_principal['C2'], color ='r');
b = pplt.scatter(M_principal['C1'], M_principal['C2'], color ='b');

pplt.figure(figsize =(9, 9))
pplt.scatter(M_principal['C1'], M_principal['C2'], c = cvec)
pplt.legend((g, k, r, b), ('Label M.0', 'Label M.1', 'Label M.2', 'Label M.-1'))
pplt.show()

dts = DBSCAN(eps = 0.0375, min_samples = 50).fit(M_principal)

labeling = dts.labels_

colours1 = {}
colours1[0] = 'r'
colours1[1] = 'g'
colours1[2] = 'b'
colours1[3] = 'c'
colours1[4] = 'y'
colours1[5] = 'm'
colours1[-1] = 'k'

cvec = [colours1[label] for label in labeling]
colors = ['r', 'g', 'b', 'c', 'y', 'm', 'k' ]
r = pplt.scatter(
M_principal['C1'], M_principal['C2'], marker ='o', color = colors[0])
g = pplt.scatter(
M_principal['C1'], M_principal['C2'], marker ='o', color = colors[1])
b = pplt.scatter(
M_principal['C1'], M_principal['C2'], marker ='o', color = colors[2])
c = pplt.scatter(
M_principal['C1'], M_principal['C2'], marker ='o', color = colors[3])
y = pplt.scatter(
M_principal['C1'], M_principal['C2'], marker ='o', color = colors[4])
m = pplt.scatter(
M_principal['C1'], M_principal['C2'], marker ='o', color = colors[5])
k = pplt.scatter(
M_principal['C1'], M_principal['C2'], marker ='o', color = colors[6])

pplt.figure(figsize =(9, 9))
pplt.scatter(M_principal['C1'], M_principal['C2'], c = cvec)
pplt.legend((r, g, b, c, y, m, k), 
            ('Label M.0', 'Label M.1', 'Label M.2', 'Label M.3', 'Label M.4','Label M.5', 'Label M.-1'),
            scatterpoints = 1,
            loc ='upper left', 
            ncol = 3,
            fontsize = 10) 
pplt.show()