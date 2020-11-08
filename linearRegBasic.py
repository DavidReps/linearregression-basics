
import numpy
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


# Work in Python. Load the regression data set found in the regressionData2.txt
# file into a numpy array, using numpy.loadtxt (specify ’,’ as the delimiter). The target
# attribute occupies the last column. Once loaded into D, you can extract the last column
# by writing D[:,-1]. Likewise, the other columns can be extracted as D[:,:-1]. Study
# the online documentation for all of the relevant software needed for this task.
# (a) First, train a linear regression model from the linear_model.LinearRegression
# class in sklearn on the data set. Report the model’s goodness of fit by using the
# score function (instance method) of the model, using the full data set for scoring.
# (b) Next, use the cross_val_score function in the sklearn.model_selection module to evaluate the goodness of fit of the sklearn.linear_model.LinearRegression
# model type differently, using 4-fold cross-validation. Report the results. How do
# the two goodness of fit values compare? Submit your source code and the results.


# 5. In this task, you will study one example of the relationship between model complexity
# and generalization performance. Use the same data set as in the preceding task, but
# consider a sequence of 20 versions of it. All of the versions use all of the data examples
# (rows). However, for each i, the i-th version, Di
# , uses only the first i predictive (nontarget) attributes (columns). You can generate these various versions by appropriate
# indexing in Python / numpy. For example, for the first 4 columns of D, write D[:,:4].
# As in the preceding task, use the LinearRegression class to construct all models.
# (a) For each i, compute two goodness of fit values: (i) the value returned by the
# score function of a model trained on Di
# if the full Di
# is provided as the test data
# input to score, and (ii) the mean of the values returned by 4-fold cross-validation
# over Di using model_selection.cross_val_score (refer to the preceding task).
# Plot the two sets of scores as functions of i using matplotlib.pyplot.plot.
# (b) Discuss your results from the preceding part. Do the two sets of scores behave
# the same, or differently? Why? Please refer to details of the results, as needed,
# some of which may not be apparent in the plot


D = numpy.loadtxt('regressionData2.txt', delimiter = ',')

LinearR = []
CrossVal = []


y = D[:,-1]
X = D[:,:-1]

#intitialize linear regression
model = LinearRegression()

#train the model
model.fit(X,y)

#score the model
print("Linear Regression goodness of fit:")
print(model.score(X, y), '\n')

#cross val scores
print("Cross Validation goodness of fit value:")
print(numpy.mean(cross_val_score(model, X, y, cv=4)), '\n')


for i in range(20):

    X = D[:,:i+1]

    model = LinearRegression()
    model.fit(X,y)
    temp = model.score(X, y)

    LinearR.append(temp)
    CrossVal.append(numpy.mean(cross_val_score(model, X, y, cv=4)))


plt.title("Linear Regression:")
plt.ylabel("Accuracy")
plt.xlabel("Test number")

plt.plot(LinearR, color = 'green')
plt.show()

plt.title("Cross Validation:")
plt.ylabel("Accuracy")
plt.xlabel("Test number")
plt.plot(CrossVal)
plt.show()
print(LinearR, '\n')
print(CrossVal, '\n')
