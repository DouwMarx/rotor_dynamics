from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
import numpy as np

x_range = np.linspace(0,1,1000)

# Define a model that generates data
def model(theta):
    r = np.sin(100*theta[0]*x_range)
    return r


problem_dimensionality = 1
number_of_design_points = 500

# Create datapoints, later with lhs
design_space = np.random.random((number_of_design_points,problem_dimensionality))

# Compute the response for a certain designpoint
#data = np.array([model(design) for design in design_space])


test_percentage = 20
n_training_samples = int(number_of_design_points*(100 - test_percentage)/100)
# Split into training and test data
indices = np.random.permutation(number_of_design_points)
training_idx, test_idx = indices[:n_training_samples], indices[n_training_samples:]
training_designs, test_designs = design_space[training_idx,:], design_space[test_idx,:]

training_response = np.array([model(design) for design in training_designs])
test_response = np.array([model(design) for design in test_designs])


# Plot the training and test set
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection='3d')
# Plot training
for response in enumerate(training_response):
    ax.plot(x_range, training_designs[response[0]]*np.ones(len(x_range)), response[1],c = "green", label = "train")

# Plot test
for response in enumerate(test_response):
    ax.plot(x_range, test_designs[response[0]]*np.ones(len(x_range)), response[1],c = "red",label = "test")
ax.legend()

#Code from Stackoverflow https://stackoverflow.com/questions/26337493/pyplot-combine-multiple-line-labels-in-legend to
# use same legend for different plots
handles, labels = plt.gca().get_legend_handles_labels()
newLabels, newHandles = [], []
for handle, label in zip(handles, labels):
  if label not in newLabels:
    newLabels.append(label)
    newHandles.append(handle)

plt.legend(newHandles, newLabels)
plt.show()

# Train a regression model
from sklearn.linear_model import LinearRegression

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import  RBF, ConstantKernel as C

# define model
epsilon = 1e-2 # Scale parameter for RBF # Smaller = less pointy, more smooth
kernel = C(1.0, (1e-3, 1e3)) * RBF(epsilon, (1e-6, 1e2))
model = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=20)
# fit model
model.fit(training_designs, training_response)

yhat = model.predict(training_designs)
# summarize prediction


# Plot learned response surface
x = x_range
y_range_plot = np.linspace(0,1,1000)
y = np.array([y_range_plot]).T # design parameter
X, Y = np.meshgrid(x, y)
Z = model.predict(y)

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z,alpha = 0.3)

ax.set_xlabel('x')
ax.set_ylabel('theta')
ax.set_zlabel('response')

plt.show()

# Plot surface on its own
fig = plt.figure()
ax = fig.gca(projection='3d')
x = x_range
y = np.array([np.linspace(0,1,1000)]).T # design parameter
X, Y = np.meshgrid(x, y)
Z = model.predict(y)

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z,alpha = 0.3)

ax.set_xlabel('x')
ax.set_ylabel('theta')
ax.set_zlabel('response')

plt.show()
