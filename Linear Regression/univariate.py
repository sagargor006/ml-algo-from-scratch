import numpy as np
import math

def load_data(training_data=float):
    data = np.genfromtxt('salary.csv', delimiter=',', skip_header=1)

    # Shuffle the data randomly
    np.random.seed(42)  # for reproducibility (optional)
    np.random.shuffle(data)
    
    split = int(len(data) * training_data)
    train_x = data[:split,0]
    train_y = data[:split,1]

    test_x = data[split:,0]
    test_y = data[split:,1]
    return train_x,train_y,test_x,test_y

def z_score_normalized_features(x):
    mu = np.mean(x)
    sigma = np.std(x)
    return (x - mu) / sigma

def z_score_denormalized_features(x,normalized_x):
    mu = np.mean(x)
    sigma = np.std(x)
    return (normalized_x * sigma) + mu

def predict(x,w,b):
    return w * x + b

def calculate_cost(x,y,w,b):
    m = x.shape[0]
    total_cost = 0

    for i in range(m):
        predicted_y = predict(x[i],w,b)
        total_cost += (predicted_y - y[i]) ** 2

    return (1 / (2*m))* total_cost

def find_gradient_derivatives(x,y,w,b):
    m = x.shape[0]
    dj_dw = 0.
    dj_db = 0.
    for i in range(m):
        predicted_y = predict(x[i],w,b)
        temp_dj_dw = (predicted_y - y[i])*x[i]
        temp_dj_db = (predicted_y - y[i])

        dj_dw += temp_dj_dw
        dj_db += temp_dj_db

    dj_dw = (1/m)*dj_dw
    dj_db = (1/m)*dj_db

    return dj_dw,dj_db

def calculate_gradient_descent(x_in,y_in,w,b,alpha=0.001,iterations=1000):
    w_histort = []
    w_history = []
    cost_history = []

    for i in range(iterations):
        dj_dw ,dj_db = find_gradient_derivatives(x_in,y_in,w,b)
        #print(f"dj_dw : {dj_dw} , dj_db : {dj_db}")
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
    
        if i <100000:
            cost_history.append(calculate_cost(x_in,y_in,w,b))
            w_history.append(w)
        
        if i% math.ceil(iterations/10)==0:
            w_histort.append(w)
            print(f"Iterations : {i} Cost : {cost_history[-1]:0.2e}")

        if i==iterations-1:
            w_histort.append(w)
            print(f"Iterations : {i+1} Cost : {cost_history[-1]:0.2e}")

    return w,b,w_history,cost_history