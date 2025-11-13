import pandas as sg
import matplotlib.pyplot as rk
from sklearn.cluster import KMeans
rates = sg.read_csv('data.csv')
print(rates)
filter = rates[['total_bill','tip']]
print(filter)
KMeans = KMeans(n_clusters=5,random_state=49)
KMeans.fit(filter)
filter['cluster']=KMeans.labels_
rk.figure(figsize=(8,8))
rk.scatter(filter['total_bill'],filter['tip'],c=filter['cluster'],cmap='cool',s=100,label="bills")
centroids = KMeans.cluster_centers_
rk.scatter(centroids[:,0],centroids[:,1],c='black',s=200,marker='x',label='centroids')
rk.title('Total Sales In The Past Days')
rk.xlabel('Bills')
rk.ylabel('Tips')
rk.show()
