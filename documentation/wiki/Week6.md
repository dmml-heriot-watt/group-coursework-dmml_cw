## Week- 6 Task
- [x] Meet Team Member 
- [x] Update GitHub Repository 
- [x] Explore K-Mean clustering

### Clustering 
Implemented homogeneity score and completeness score in K-Means Clustering, utilizing the Elbow method, silhouette score of previous implimentation of K-Mean clustering , we moved to Feature selection for getting batter quality of cluster so, Feature selection of pH and alcohol was performed based on a correlation matrix and pair plot analysis. After feature selection we perform 3 cluster using K-Means clustring algorithm.

By seeing elbow and silhouette score of this clustering algorithm, we can say that feature selection significantly improved the accuracy of the clustering model.

**Click to view**
 - [K Means Algorithm with score](https://github.com/dmml-heriot-watt/group-coursework-dmml_cw/blob/main/notebooks/K_Means_Clustering_withScore.ipynb) 
- [EM Algorithm]( https://github.com/dmml-heriot-watt/group-coursework-dmml_cw/blob/main/notebooks/EM.ipynb)

#### Results
to evaluate accuracy of clustering algorithm in wine dataset for quality prediction we also calculated homogeneity score and completeness score of each algorithm for different number of cluster.

The following are the key results of the clustering analysis:

The following **Score Table** shows the key results of the K-Means clustering analysis:
| Clustering Technique                        | Homogeneity Score | Completeness Score | Optimal k (Elbow Method) | Average Silhouette Score |
|--------------------------------------------|-------------------|--------------------|--------------------------|---------------------------|
| 9-cluster K-means Clustering (All Features) | 0.41              | 0.24               | Not Clear                | 0.24537          |
| 4-cluster K-means Clustering (All Features) | 0.1857            | 0.0974             | Not Clear                | 0.2453                    |
| 3-cluster K-means Clustering (Selected Features) | 0.1448       | 0.0398             | 3                        | 0.85                      |

- By getting the Elbow method and silhouette score for **3-cluster K-means Clustering (Selected Features)** we can say that, feature selection significantly helped to improved the accuracy of the clustering model. but 0.14 Homogeneity score and 0.0398 Completeness score indicates that the clustering algorithm may not be effectively capturing the wine quality in the dataset.
