## Here we're implementing the gradient descent by scratch [model.fit() function]
## Expected answer. m = 0.05168176, b=18.0465

## Importing necessary libraries
import pandas as pd
import numpy as np

def gradient_descent(x, y, learning_rate = 0.01, epochs = 3000):
    """
    Implementing gradient descent from scratch.
    
    Parameters:
        x : Input feature values
        y : Target values
        learning_rate : Step size for updates
        epochs : Number of iterations for training
    
    Returns:
        (b_original, m_original): Intercept and slope scaled back to original range
    """
    
    ## Initializing slope (m) and intercept (b)
    m, b = 0.0, 0.0
    
    
    ## Min-Max Scaling for stability
    
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    x_scaled = (x - x_min) / (x_max - x_min)
    y_scaled = (y - y_min) / (y_max - y_min)
    
    ## Gradient Descent Iterations
    for epoch in range (epochs+1):
        
        ## Predicting value using current parameters
        y_pred = m*x_scaled + b
        
        ## Computing error and cost (Mean Squared Error)
        error = y_scaled - y_pred
        cost = np.mean(error**2)
        
        ## Computing gradients
        dm = -2*np.mean(error *  x_scaled)
        db = -2*np.mean(error)
        
        ## Updating parameters
        b = b - db * learning_rate
        m = m - dm * learning_rate
        
        ## Printing the progress after every 100 epochs
        if epoch%100 == 0:
            print(f" m = {m}, b = {b}, Epoch = {epoch}, Cost = {cost}")
            
    # Scale back the coefficients to original scale
    b_original = b * (y_max - y_min) + y_min - m * (y_max - y_min) * x_min / (x_max - x_min)
    m_original = m * (y_max - y_min) / (x_max - x_min)

    return b_original, m_original
        
    
        
    
if __name__ == "__main__":
    # x = np.array([1, 2, 3, 4, 5])
    # y = np.array([5, 7, 9, 11, 13])
    df = pd.read_csv("home_prices.csv")
    
    x = df["area_sqr_ft"].to_numpy()
    y = df["price_lakhs"].to_numpy()
    
    ## These two columns have are on a different scale so we'll have to normalize it
    b, m = gradient_descent(x, y)
    
    print(f"\nFinal result:\nm = {m}\nb = {b}")
 

