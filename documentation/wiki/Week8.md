## Week- 8 Task
- [x] Meet Team Member 
- [x] Update GitHub Repository 
- [x] Random Forest

### Random Forest
A single tree reached an accuracy of 0.52 for the Wine Quality prediction. This accuracy was obtained thanks to a 23 maximum length Decison Tree, while with a maximum length of 3 the. accuracy is 0.54. The tree seemed to overfit. 
Another model that derives from Decision Trees is the Random Forest. A Random Forest is a method that combines multiple decision trees to enhance predictive accuracy and generalization. It also mitigates overfitting. The experiment was made using all the data as input (except wine type), the target being the Wine Quality. The ouput labels are the predicted Quality. 

**Click to view :**
- [Random Forest Notebook](https://github.com/dmml-heriot-watt/group-coursework-dmml_cw/blob/main/notebooks/DecisionTreeQuality.ipynb)
  
#### Results
The following **Performance Metrics Table** shows the results of the Decision tree analysis:
| Model                        | Accuracy Score | Precision | Recall | F1-score |
|------------------------------|-----------------|-----------|--------|----------|
| Wine Quality(classification)        |      0.52      |    0.63    |   0.52  |   0.45    |

The following **Performance Metrics Table** shows the results of the Random Forest analysis:
| Model                        | Accuracy Score | Precision | Recall | F1-score |
|------------------------------|-----------------|-----------|--------|----------|
| Classification        |      0.626      |    0.6159    |   0.626  |   0.607    |

This data set seems complex and a decision tree based model is limited for the quality prediction. Even a random forest can only take the accuracy from 0.58 to 0.62 with 10 trees, almost the same with 1000 trees. The Random Forest gives then better results without being great for quality prediction. The supervised models for this task are however more suited for Quality prediction than clustering. 
