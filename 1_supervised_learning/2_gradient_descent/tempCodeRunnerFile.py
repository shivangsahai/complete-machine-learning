import pandas as pd
import numpy as np

# we're implementing the model.fit() function
# Expected answer. m = 0.05168176, b=18.0465

def gradient_descent(x, y, learning_rate = 0.1, epochs = 1000):
    m, b = 0, 0
    
    
    for epoch in range (epochs):
        y_pred = m*x + b
        error = y - y_pred
        cost = np.mean(error**2)
        
        dm = -2*np.mean(x * error)
        db = -2*np.mean(error)
        
        b = b - db * learning_rate
        m = m - dm * learning_rate
        
        if epoch%50 == 0:
            print(f" m = {m}, b = {b}, Epoch = {epochs}, Cost = {cost}")
        
    
if __name__ == "__main__":
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([5, 7, 7, 11, 13])
    gradient_descent(x, y)
 

