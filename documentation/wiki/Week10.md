## Week- 10 Task
- [x] Meet Team Member 
- [x] Update GitHub Repository 
- [x] Implement CNN

### Neural Networks

After implementation of ANN on Fruit-360 dataset we tried to implement CNN on that dataset.

### Experiment

**Click to view :**  [Neural Network Implimentation Notebook](https://github.com/dmml-heriot-watt/group-coursework-dmml_cw/blob/main/notebooks/DeepLearning2.ipynb)

-  **CNN (Convolutional Neural Network):** We utilized Convolutional layers with MaxPooling, followed by Dense layers and an output layer with softmax activation. Adam was used as the optimizer for the hyperparameters. To assess the model's accuracy, agin we used the accuracy score metric, and categorical crossentropy selected as the loss function.

#### Results
- To get the accuracy of all neural networks, we calculated Precision, Recall and F1-score and also compare the result of that.

The following  **Performance Metrics Table**  shows the results of the Neural Network analysis:


| Model                           | Accuracy Score | Loss   | Precision | Recall | F1-score |
|---------------------------------|-----------------|--------|-----------|--------|----------|
| ANN with single Layer           | 0.84            | 3.2849 | 0.89      | 0.84   | 0.84     |
| ANN with three hidden Layers    | 0.9199          | 0.4830 | 0.92      | 0.92   | 0.91     |
| CNN                             | 0.96            | 0.2222 | 0.97      | 0.96   | 0.96     |


- By seeing the result, we can say CNN perform extremely batter then other Neural Network on Fruit-360 dataset.

#### Reference
- to complete this task we take a reference from following:
	- [code basics](https://www.youtube.com/watch?v=Mubj_fqiAv8&list=PLeo1K3hjS3uu7CxAacxVndI4bE_o3BDtO)



