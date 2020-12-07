# Stack overflow tag predictor
Final project CS-725 Stack-overflow-tag-predictor
#### Team Name- COT
#### Team Members- Nilesh Tanwar (203050060), Ankit Kumar (203050109), Shivam Dixit (193050012)
The task is to classify the questions that appeared at some time on stack overflow and assign them suitable tags so that a particular target audience corresponding to that tag can be catered.

### Steps involved -
1. Acquiring the data. Fortunately kaggle has a datset of 10% of stack overflow questions. Link- https://www.kaggle.com/stackoverflow/stacksample
2. Identifying some data patterns to show some relation and statistics in the dataset.
3. Obtain final cleaned data on which different machine learning techniques can be applied.
4. Apply different machine learning techniques and obtain suitable metrics to compare efficiency of different methods.

Dataset taken from kaggle : https://www.kaggle.com/stackoverflow/stacksample

### Approach-1
1. Data Cleaning task.
2. Modelling words in terms of frequency and obtaining a suitable inference from it.
3. Training using sklearn libraries and comparing the accuracy.

### Approach-2
1. Data Cleaning task (somewhat similar to approach-1), also made use of BeautifulSoup library to get text out from HTML elements (present in Question Body)
2. Drawing inference from data patterns and vectorization of text data (using tf-id).
3. Merging of Tags.csv and Question.csv on ‘Id’ field to form a suitable data frame which models the dependencies in a better way.
4. Training using sklearn libraries and comparing the accuracy to find a suitable fit. Training was done on similar models as were mentioned in approach-1.

### Conclusion
We have measured the accuracy and time of execution (on test set) for each model in each approach and landed on some common grounds : 
1. On one side where Ridge classifier (82% accurate) beats all in terms of accuracy, Naïve Bayes (62% accuracy) classifier can be good choice if we instant execution with moderate accuracy. 
2. Moreover the SGDClassifer (Linear SVM- 80% accuracy) approximates the results of Ridge Classifier.
3. The assumption of Naïve Bayes did not hit the accuracy of model that bad and showed quick and accurate results.

But as a final recommendation we would suggest the use of Ridge classifier for tag prediction task due to its high accuracy and highly accurate tag prediction.

### Further work
Although we have worked out 2 different approaches to classify the data but still there is a huge scope of further exploring the data and coming up with a more accurate model like :
1. The Answers.csv file can be used to predict the context of the answer which in turn can predict the tag for that question Id.
2. Also the change in popularity of certain tags can be modelled ad this data can also provide a different dimension to the analysis.
3. We can build a neural network to predict a suitable tag that can implicitly learn some new data patterns that can help to study the prediction procedure in a different domain and lots more…

#### Team name - COT
