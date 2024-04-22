import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 2: Prepare the data
def fetch_covid_data():
    url = "https://disease.sh/v3/covid-19/historical/all?lastdays=365"
    response = requests.get(url)
    data = response.json()
    x = []
    y = []
    for i, cases in enumerate(data['cases'].values()):
        x.append(i+1)
        y.append(cases)
    return x, y

# Fetch real-time COVID-19 data
x, y = fetch_covid_data()

# Step 3: Create and fit the linear regression model
model = LinearRegression()
model.fit(np.array(x).reshape(-1, 1), np.array(y))

# Step 4: Predict the values using the linear regression model
x = np.array(x).reshape(-1, 1)  # Reshape x again before prediction
y_pred = model.predict(x)

# Step 5: Plot the data and the linear regression line
plt.scatter(x, y, color='blue', label='Actual Data')
plt.plot(x, y_pred, color='red', label='Linear Regression Line')
plt.xlabel('Number of Days')
plt.ylabel('Number of Cases')
plt.title('COVID-19 Analysis of One Year (365 days)')
plt.legend()
plt.show()
