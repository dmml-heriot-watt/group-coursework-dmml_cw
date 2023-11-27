## Week- 7 Task
- [x] Meet Team Member 
- [x] Update GitHub Repository 
- [x] Implement Decision Tree



### Decision Tree 
In Wine dataset, each row in the dataset represents a specific wine sample, and the features serve as inputs for the decision trees. We have implemented two decions trees, the output labels represent the wine type and quality of the wine respectively for these trees. Also random forest has been implemented for wine quality as output label.

**Click to view :**
  - [Decision Tree Classification Notebook](https://github.com/dmml-heriot-watt/group-coursework-dmml_cw/blob/main/notebooks/DecisionTreePara.ipynb) 

#### Experiment

**Decision Tree Classification:**
In our exploration of decision tree models for wine type and wine quality prediction, we started with all available features. The initial decision tree showed impressive accuracy, reaching 99.97% on the training set and 99.33% on the test set. However, we were worried that the model might be overfitting to the training data. We tried to address this by using K-fold cross-validation and adjusting the parameters of the tree. To evaluate the model's accuracy, we used the accuracy, F1-score, Precision, Recall etc.
 	
#### Results

The following **Performance Metrics Table** shows the results of the **Decision tree** analysis:
| Model                          | Accuracy Score | Precision | Recall | F1-score |
| ------------------------------ | -------------- | --------- | ------ | -------- |
| Wine Type(classification)      | 0.95           | 0.96      | 0.95   | 0.95     |
| Wine Quality(classification)   | 0.52           | 0.63      | 0.52   | 0.45     |
	

