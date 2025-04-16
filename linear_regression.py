import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normal (g):
    return (g - g.min()) / (g.max() - g.min())

def hypothesisfun(x, t):
    return np.dot(x, t)

def costfun(datas, performance_index, thetas):
    res = hypothesisfun(datas, thetas)
    res = np.subtract(res, performance_index)**2
    res = res.sum()
    res = res / datas.shape[0]
    return res

def gradientDescent(datas, performance_index, learnin_rate, thetas):
    m = len(datas)
    res = hypothesisfun(datas, thetas) - performance_index
    gradient = np.dot(datas.T, res) / m
    thetas = thetas - learnin_rate * gradient
    return thetas

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

datas = pd.read_csv('Train_Data.csv')

actual_datas = pd.read_csv('Test_Data.csv')

datas.loc[datas["Extracurricular Activities"] == "Yes", "Extracurricular Activities"] = 1
datas.loc[datas["Extracurricular Activities"] == "No", "Extracurricular Activities"] = 0

hours_studied = pd.DataFrame(datas, columns=["Hours Studied"])
previous_scores = pd.DataFrame(datas, columns=["Previous Scores"])
extracurricular_activities = pd.DataFrame(datas, columns=["Extracurricular Activities"])
sleep_hours = pd.DataFrame(datas, columns=["Sleep Hours"])
sample_question_papers_practiced = pd.DataFrame(datas, columns=["Sample Question Papers Practiced"])
performance_index = pd.DataFrame(datas, columns=["Performance Index"]).to_numpy()

datas["Hours Studied"] = normal(hours_studied)
datas["Previous Scores"] = normal(previous_scores)
datas["Extracurricular Activities"] = normal(extracurricular_activities)
datas["Sleep Hours"] = normal(sleep_hours)
datas["Sample Question Papers Practiced"] = normal(sample_question_papers_practiced)

datas = datas.iloc[:, :5].to_numpy()
 
thetas = np.zeros(datas.shape[1])   

thetas = pd.DataFrame(thetas)

costs = []

for i in range(1000):
    costs.append(costfun(datas, performance_index, thetas))
    thetas_new = gradientDescent(datas, performance_index, 1 , thetas)
    thetas = thetas_new
    
costs = np.array(costs)

predictions = hypothesisfun(datas, thetas)

r2 = r_squared(performance_index, predictions) * 100
print(f"R² (Coefficient of Determination): {r2:.4f}")
    
plt.plot(costs)
plt.show()