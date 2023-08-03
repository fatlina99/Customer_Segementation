# Customer_Segementation
Customer segmentation is a process to dividing customers into groups which possess 
common characteristics based on their age, gender, profession as well as interests. 
By doing so will enable the company to gain insights of customer’s needs or 
preferences, therefore the company will design a marketing strategy to target the 
most profitable segments. 
Recently, there has been a decline in revenue for the bank and investigation has been 
carried out to identify the root of the problem. The bank found out that the reduce in 
revenue is mainly due to decrement of the deposition of money into the bank by the 
clients. Hence, the bank decided to conduct marketing campaigns to persuade more 
clients to deposit money into the bank. 
The purpose of marketing campaign is to collect customer’s needs and overall 
satisfaction. There are a few essential aspects of the marketing campaign namely, 
customer segmentation, promotional strategy, and etc. Correctly identified strategy 
may help to expand and grow the bank’s revenue. 
You are provided a dataset containing details of marketing campaigns done via 
phone with various details for customers such as demographics, last campaign 
details etc. Hence, your job as data analysts cum deep learning engineer is to 
develop a deep learning model to predict the outcome of the campaign. 
The criteria of the project are as follows:

1) Develop a deep learning model using TensorFlow which only comprises of
Dense, Dropout, and Batch Normalization layers.
2) Display the training loss and accuracy on TensorBoard
3) Create modules (classes) for repeated functions to ease your training and 
testing process

## The architecture of the model
![model_architecture](https://github.com/fatlina99/Customer_Segementation/assets/141213373/ba4844f2-cae5-4c00-be1c-706536cca05a)

## F1 Score Comparison
![f1_score_comparison](https://github.com/fatlina99/Customer_Segementation/assets/141213373/b093dbbc-694c-497c-9b81-6fe92a8d5d7b)


## Conclusion
1. The Deep Learning Model has the lowest F1-score = 0.0851. This indicates that the model is not performing well on the classification task, and its predictions are not precise or recall-friendly.

2. The KNN Model has a higher F1-score =0 .4823 compared to the Deep Learning Model. However, it is still relatively low, suggesting that the KNN model is better than the Deep Learning Model but still needs improvement.

3. The Gradient Boosting Model has the highest F1-score =0.5215 among the three models. This indicates that the Gradient Boosting model is performing the best in terms of precision and recall.

## Credit/cite
https://www.kaggle.com/datasets/kunalgupta2616/hackerearth-customer-segmentation-hackathon
