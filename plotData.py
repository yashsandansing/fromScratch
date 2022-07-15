import matplotlib.pyplot as plt
import numpy as np

class plotFigures:
    
    def scatterPlot(self, X, y):
        '''
        Plots the distribution of data in a scatter plot
        Valid for 2 classes only
        '''
        plt.figure(figsize=(10, 6))
        plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='b', label='0')
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='r', label='1')
        plt.legend();

        return plt
    
    def classificationBoundary(self, X, y, theta):
        '''
        A function that plots the figure with boundary(may need to change the limits in the function itself)
        Inputs: 
        X - input data
        y - feature
        theta - parameters learned through training

        outputs - None
        '''
        plt.figure(figsize=(10, 6))
        plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='b', label='0')
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='r', label='1')
        plt.legend();
        
        ax = plt.gca()
        ax.autoscale(False)
        theta_ = theta[:,0]  # Make theta a 1-d array.
#         x_ = np.linspace(-4, 5, 50)#play with this feature to get the appropriate limits (zoom in/out)
        x_ = np.array(ax.get_xlim())
        y_ = -(theta_[0] + theta_[1]*x_)/theta_[2]
        plt.plot(x_, y_)

        return plt