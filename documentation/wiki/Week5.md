## Week- 5 Task
- [x] Meet Team Member 
- [x] Update GitHub Repository 
- [x] Explore Dataset
- [x] Clustering



### Clustering 
Implemented K-Means Clustering and EM algorithm, utilizing the Elbow method, silhouette score, and joinplots to select efficient and correlated features for observing the means of each cluster for interpretabilityfor improved accuracy.

**Click to view**
 - [K Means Algorithm ](https://github.com/dmml-heriot-watt/group-coursework-dmml_cw/blob/main/notebooks/K_Means_Clustering.ipynb) 
- [EM Algorithm]( https://github.com/dmml-heriot-watt/group-coursework-dmml_cw/blob/main/notebooks/EM.ipynb)

#### Experiment

- In **K-Means algorithm** we implimented 9 cluster including all the feature of the dataset as we want to predict the quality of wine. Quallity have 0 to 9 value which indicates quality of wine. so, initially we selected 9 number of cluster. we also calculated Elbow method and silhouette score for getting optimum number of cluster. After getting 4 number of cluster from silhouette  score of initial clustering implimentation we perform same clustering using 4 number of cluster.
- In **EM algo** we observed joinplots to select features efficiently using Corelation Matrix.
 	
#### Results
The following are the key results of the clustering analysis:

- **K Mean:** After refining with the Elbow method and silhouette score, the optimal number of clusters was determined to be 4 for all the feature of the dataset, balancing accuracy. Need to explore more.

- **EM algo:** Observed joinplots for 'fixed acidity' vs. 'pH,' 'fixed acidity' vs. 'alcohol,' 'pH' vs. 'alcohol,' and 'sulphates' vs. 'alcohol.' The selected features were further validated by adding the distributions of quality features.
	

