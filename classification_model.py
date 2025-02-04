import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
iris_dataset = load_iris()
# Split dataset into 75% training set and 25% testing set
X_train, X_test, y_train, y_test = train_test_split(
        iris_dataset['data'], iris_dataset['target'], test_size=0.25, random_state=0)

## We can use k-nearest neighbors classifier:
# Neighbors-based classification algorithms are a type of instance-based learning,
# they don't construct a general model, but store instances of the training data.
# Classification is computed from majority vote of the nearest neighbors of each point.

# Thus building this model only consists of storing the training set,
# so when a prediction is made the algorithm finds the point in the training set closest to the new point.
n_neighbors=1
knn = KNeighborsClassifier(n_neighbors=1) # knn becomes an object that contains the algorithm.
knn.fit(X_train[:,2:], y_train) # Using only last two features for better visualization (Petal length (cm), Petal width (cm))


# Decision surface is a boundary that separates different classes in the dataset.
# It helps us to understand which areas of the graph belong to which class based on our model's predictions.
x_min, x_max = X_train[:,2].min() - 1, X_train[:,2].max() + 1  # Find range for x-axis (petal length)
y_min, y_max = X_train[:,3].min() - 1, X_train[:,3].max() + 1  # Find range for the y-axis (petal width)

h = 0.02  # Step size for the mesh grid (defines resolution for the grid)

# Create a mesh grid of points
xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h)) # xx and yy to define the grid of points covering the feature space
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()]) # Predict the class for each point in the mesh grid
Z = Z.reshape(xx.shape) # Reshape the predictions to match the shape of the mesh grid

cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])
cmap_light=ListedColormap(['orange', 'cyan', 'cornflowerblue'])

fig = plt.figure()
ax1 = fig.add_subplot(111) # Create a subplot in the figure with 1 row, 1 col, and occupying the 1st position in the grid

ax1.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='gouraud') # Uses the mesh grid to visualize decision regions

for target in iris_dataset.target_names:
    index=np.where(iris_dataset.target_names==target)[0][0] # Find the index corresponding to the target class
    ax1.scatter(X_train[:,2][y_train==index],X_train[:,3][y_train==index], # Plot the training points for each class
                cmap=cmap_bold, edgecolor='k', s=20, label=target)

ax1.set_xlim(x_min,x_max) # x-axis limits
ax1.set_ylim(y_min,y_max) # y-axis limits

ax1.legend() # Add legend to indicate class labels
ax1.set_xlabel("petal length (cm)")
ax1.set_ylabel("petal width (cm)")
ax1.set_title("3-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, 'uniform'))

plt.show() # This shows us where new datapoints would be classified based on the petal length \ width


# Predict the class labels for the test set
y_pred = knn.predict(X_test[:,2:])  # Predict
print("Test set predictions:\n {}".format(y_pred))  # Print predicted labels for test data

# Calculate / print accuracy of the model on the test set
print("Test set score: {:.2f}".format(knn.score(X_test[:,2:], y_test)))  # Print accuracy score


""" # Uncomment to see how we can also make a new datapoint and test the trained KNN model with it
new_data=np.array([[4,3.5,1.2,0.5]]) # New data point
prediction = knn.predict(new_data[:,2:])  # Use only the last two features, same decision boundaries as above

print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(iris_dataset['target_names'][prediction]))  # Output the predicted species name
"""
