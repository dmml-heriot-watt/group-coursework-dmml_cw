## Week- 7 Task
- [x] Meet Team Member 
- [x] Update GitHub Repository 
- [x] Implement Decision Tree



### Clustering 
For implementation of Decision tree we have selected Wine Quality Dataset that contains information about the chemical properties of various wines. Each row in the dataset represents a specific wine sample, and the features serve as inputs for the decision trees.The output labels represent the quality of the wine. Specifically the labels are assigned based on the quality rating of each wine.

**Click to view :**
  - [Decision Tree Classification Notebook](https://github.com/dmml-heriot-watt/group-coursework-dmml_cw/blob/main/notebooks/DecisionTreePara.ipynb) 
  - [Random Forest Notebook](https://github.com/dmml-heriot-watt/group-coursework-dmml_cw/blob/main/notebooks/DecisionTreeQuality.ipynb)

#### Experiment

**CART Algorithm:**
    In our exploration of decision tree models for wine quality prediction, we initiated with a comprehensive analysis using all available features. The initial decision tree showed impressive accuracy,     
    reaching 99.96% on the training set and 99.20% on the test set. However, we worried that the model might be overfitting to the training data, leading to poor performance on new data. We tried to address     
    this by using K-fold cross-validation and adjusting the complexity of the tree. To address concerns about the model's complexity, we tuned the model's parameters. To evaluate the model's accuracy, we used the accuracy score metric.
 	
#### Results
The following **Performance Metrics Table** shows the results of the Decision tree analysis:
| Model                        | Accuracy Score | Precision | Recall | F1-score |
|------------------------------|-----------------|-----------|--------|----------|
| Classification        |      0.58      |    0.57    |   0.58  |   0.57    |
	

