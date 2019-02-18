from copy import deepcopy
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

class CookDist(object):
    """
    Computes the Cook Distance for a predictor and a dataset. 
    cook_dist_instance computes the Cook Distance for a given instance. 
    cook_dist computes the Cook distance for all instances in dataset. 
    
    Example:
    cookdist = CookDist(predictor = RandomForestRegressor()) 
    cd = cookdist.cook_dist(X, y) #input X needs to be a pandas dataframe
    #output has same index as X
    """
    def __init__(self, predictor= RandomForestRegressor()):
        self.predictor = predictor 
        
    def fit(self, X, y):
        self.predictor.fit(X, y)
        self.prediction = self.predictor.predict(X)
        
    def cook_dist_instance(self, X, y, i):
        predictor_i = deepcopy(self.predictor)
        predictor_i.fit(X.drop(i, axis=0), y.drop(i, axis=0))
        prediction_i = predictor_i.predict(X)
        return mean_squared_error(self.prediction, prediction_i)
    
    def cook_dist(self, X, y):
        self.fit(X, y)
        vect = pd.Series(X.index)
        vect.index = X.index
        cd = vect.apply(lambda i: self.cook_dist_instance(X, y, i))
        return cd
