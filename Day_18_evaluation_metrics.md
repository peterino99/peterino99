# Day 18 - Evaluation Metrics
<br>

## 4th of June 2021
<br>

![state_of_mind](https://github.com/peterino99/peterino99/blob/main/state_of_things.jpg)


<br>

*Protocol by Peter*

---

## 1. Rules of the Thumb on Models and Metrics
<br>

* Staying clear on the point of building a model: 

    * Models always need to be informed by a knowledge of why we are building them
    * evaluation? prediction? 
    * for whom? 
    * don't build models for models' sake

* Advice from the coaches:
    * Models should address product needs with as simple a structure as possible

    * differentiate between Business Metrics and Model performance 

        * Business metrics capture how well a model serves business' needs (usable for customers? readily adopted by the finance people?)

        * Model performance tells us how well the model can do without regard to its business purpose: key for us as data scientists
        * we need to be clear on whether a model is succeeding or failing
        * metrics can tell us where to improve 
        * can help us avoid misunderstanding our data

    * We do well to know how well our models can do, but often need to relate model metrics to stakeholders in ways they can understand: pure RMSE, precision, recall are usually too complex to clarify what the business side needs to know
        * creating a custom metric that the business side can understand may be a good way to proceed
<br>
## 2. Regression vs Classification
<br>

* Regression models give us values we can interpret, such as the prediction of tomorrow's temperature based on previous trends

* Classification models use a threshold to break outcomes into categories, often binary in nature and with implicit/ context-specific meaning

    * degree temperature vs. hot/cold

    * number of dark cells vs. cancerous lesion/benign growth

* we usually want models that are overly sensitive when using classification: it is better to generate false positives that can be evaluated 

* Regression models use aggregate metrics such as RMSE, RMSLE, MAPE, classification models use accuracy, precision, recall 

<br>
## 3. Metrics for classification models using a Confusion Matrix
<br> 


* A Confusion Matrix counts how often the model predicted correctly and not

    * true and false positives (false alarms, type I errors); in the general case, y = 1
    * true and false negatives (missed events, type II errors); in the general case, y = 0

    ![confusion_matrix](https://github.com/peterino99/peterino99/blob/main/confusion.jpg?raw=true)

        * left axis: actually observed outcomes
        * top axis: predicted outcomes
        * the implementation of confusion matrices in sklearn uses this convention

     
<br>
### A. Accuracy
<br>

* how often the model is right

    Accuracy = TP + TN / (TP + FP + TN + FN)

This measure can be misleading when one class is rare

        * Just stating accuracy might well be insufficient

* as an example stolen from wikipedia: an image recognition program recognizes 5 dogs in a picture containing 12 dogs and 10 cats, three it identifies as cats (these are false positives). 

        * TP is 5 (true dogs)
        * FP is 3 (cats)
        * FN is 7 (missed dogs)
        * TN is 7 (cats not counted)
        * The accuracy is 12 / 22 = 54.5% 

<br>

### B. Precision
<br>

* Precision is the ratio of TP outcomes to the total number of positive outcomes predicted by a model. The precision looks at how precise our model is as follows
<br> 

        * Precision = TP / (TP + FP) 

<br> 
* from the dog image recognition example: the program's precision is 5/8, or 62.5% (5 correct divided by 5 tp + 3 fp)  
<br>
<br>

### C. Recall
<br>

* Recall calculates what proportion of the TP outcomes our model has predicted


<br>

        * Recall = TP / (TP + FN)

* from the dog image recognition example: the recall is 5/12, or 41.6%  (5 / (5tp + 7fn))

<br>

### D. Precision-Recall Curve
 
<br>

* Plots Precision against Recall depending on the threshold in question
* represents the fact that Precision and Recall exist in a trade off relationship
* can help determine a threshold the stakeholder can agree to

    ![PRC](https://github.com/peterino99/peterino99/blob/main/PRC.jpg?raw=true)

<br>

### E. F1-score
<br>

* the harmonic mean of Precision and Recall 
* the harmonic mean punishes low rates
* F1 is a suitable measure of models tested with imbalanced datasets

        * F1-score = 2 * Precision * Recall / (Precision + Recall)

* in the dog image recognition example: F1 = 2 * .416 * .625/(.416 + .625) = .5
<br>

### E. Receiver Operating Curve (ROC) 
<br>

* visualization of True positive rate (recall) vs. false positive rate for different thresholds 

    * 45 degree line is equivalent to a model that randomly guesses yes-no (flipping a coin)

    * a perfect model produces the green edge 

    * a model with better than random performance will be above the 45 degree line 

    ![ROC](https://github.com/peterino99/peterino99/blob/main/ROC.jpg?raw=true)

### G. Area under the ROC (AUC)

<br>

* numeric representation the results from integration of the ROC
* a score of .5 represents the area under the 45 degree line
* a perfect model would produce an AOC of 1
* by the goldilocks principle, most models will be between 5. and 1

    ![AUC](https://github.com/peterino99/peterino99/blob/main/AOC.jpg?raw=true)




## 4. Implementation of classification metrics in Python
<br>

* can be obtained by using functions from sklearn.metrics module
    * from  sklearn.metrics import confusion_matrix (and so on)

confusion_matrix(df.actual_label.values, df.predicted_RF.values)

* accuracy: metrics.accuracy_score
* f1:     metrics.f1_score
* precision: metrics.precision_score
* recall: metrics.recall_score

There are many more: see: https://scikit-learn.org/stable/modules/model_evaluation.html

* we also coded functions for these values!! 

## 5. Aggregation metrics for linear regression

* these were largely familiar from our previous work on linear regression

        * Mean Absolute Error is the average of the absolute differences between predictions and actual values. It gives an idea of how wrong the predictions were. It measure gives an idea of the magnitude of the error, but no idea of the direction (e.g. over or under predicting).

        * Mean Squared Error is derived by Taking the square root of the mean squared error converts the units back to the original units of the output variable and can be meaningful for description and presentation. This is called the Root Mean Squared Error (or RMSE).

        * R^2 (R Squared) metric provides an indication of the goodness of fit of a set of predictions to the actual values. In statistical literature, this measure is called the coefficient of determination. This is a value between 0 and 1 for no-fit and perfect fit respectively.


* we also coded functions for these values!! 

* necessary to have actual and predicted values

* use sklearn approach to train data using train/ test split

![train/test](https://github.com/peterino99/peterino99/blob/main/train.jpg?raw=true)










