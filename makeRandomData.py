'''
Return a randomly generated linearly separable 
dataset for classification
'''

from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


class randomData:
    def __init__(self, 
                 n_features=2,
                 n_redundant=0):
        #initializing the features that will be fixed
        
        self.n_features=2 #setting this as 2 every time since this is a binary dataset
        self.n_redundant=0
        
       
    def createData(self,
                   n_samples=5000, 
                   n_classes=2, 
                   n_clusters_per_class=1, 
                   weights=None, 
                   flip_y=0.01, 
                   class_sep=1, 
                   hypercube=True, 
                   shift=0.0, 
                   scale=1.0, 
                   shuffle=True, 
                   random_state=None):
        
        data = make_classification(n_samples=n_samples, 
                                   n_features=self.n_features,
                                   n_redundant=self.n_redundant, 
                                   n_classes=n_classes, 
                                   n_clusters_per_class=n_clusters_per_class, 
                                   weights=weights, 
                                   flip_y=flip_y, 
                                   class_sep=class_sep, 
                                   hypercube=hypercube, 
                                   shift=shift, 
                                   scale=scale, 
                                   shuffle=shuffle, 
                                   random_state=random_state)
        
        X = data[0][:, :2]
        y = data[1]
        
        return X, y
    
    def split_data(self, X, y, test_size):
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size)
        
        return X_train, X_test, y_train, y_test                