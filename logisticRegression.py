import numpy as np
from sklearn.metrics import accuracy_score
'''
A Logistic Regression algorithm
designed for binary linear classification
'''

class LogisticRegression():
    def __init__(self, lr=0.01, num_iter=1000):
        self.lr = lr
        self.num_iter = num_iter
    
    
    def sigmoid(self,z):
        '''
        Calculate the sigmoid of the input vector
        '''

        z = 1/(1+np.exp(-z))

        return z
    
    def compute(self,X,y,theta):
        '''
        Base of the logistic regression model
        Computes the cost and the gradient
        Inputs - 

        X: features
        y: label/output
        theta: parameters

        Outputs:
        J: cost calculated
        grad: gradient to adjust parameters (move up/down)
        '''
        m=y.shape[0]

        z = np.dot(X, theta)
        hyp = self.sigmoid(z)

        J = (1/m)*sum(np.dot(-y.T, np.log(hyp)) - np.dot((1-y).T, np.log(1-hyp)))

        #y has the shape (150,)
        #need to reshape it to (150,1) to
        #calculate grad
        y=np.array(y).reshape(y.shape[0],1)
        grad = (1/m)*np.dot(X.T, (hyp-y))

        return J, grad


    def add_intercept(self,X):
        '''
        Add an intercept term to input X
        to accodomate the w0 term
        '''
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def predict(self,X, theta, y_test):
        '''
        Predict the output of the model (1/0)
        '''
        z = self.sigmoid(np.dot(X, theta))
        return (z>=0.5).astype(int).reshape(y_test.shape)
    
    
    def train(self, X, y, X_test, y_test, num_iter=1000, learning_rate=0.01):
        '''
        Main function used for training the classifier
        Returns 2 things
        1. a dictionary called "history" to be used for plotting purposes
        2. "theta" - a list of optimized parameters used for predictscraion purposes
        '''
        #initialize the parameters before starting the iteration
        #using X.shape[1] since each "feature" has it's own parameter
        #and not each input row 

        theta=np.zeros((X.shape[1]+1, 1))
        X=self.add_intercept(X)
        X_test=self.add_intercept(X_test)

        history={"cost":[], "predictions":[], "training_accuracy":[]}
        for iteration in range(num_iter):

            cost, gradient = self.compute(X, y, theta)
            theta = theta - (learning_rate*gradient)

            history["cost"].append(cost)

            trainAcc = self.predict(X, theta, y)
            history["training_accuracy"].append(accuracy_score(y, trainAcc))
            
            preds=self.predict(X_test, theta, y_test)
            history["predictions"].append(accuracy_score(y_test, preds))
        return history, theta