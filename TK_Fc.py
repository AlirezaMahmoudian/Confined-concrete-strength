#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tkinter as tk
from tkinter import ttk
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# Load data
df = pd.read_excel(r"D:\Articles\Isleem\Topic 8\DataForMLModelsTopic-8.xlsx", sheet_name='Dataset2', header=0)
X = df.iloc[:, [0, 1, 2, 3]].values
y = df.iloc[:, 4].values.reshape((-1, 1))
Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=0.7, random_state=0)

# Create Tkinter window
window = tk.Tk()
window.title("Model Prediction")

# Define models
models = {
    "DT": DecisionTreeRegressor(random_state=0),
    "RF": RandomForestRegressor(random_state=0),
    "GB": GradientBoostingRegressor(random_state=0),
    "XGB": XGBRegressor(random_state=0)
}

# Define targets
Targets = {
    "Fcc": df.iloc[:, 4].values.reshape((-1, 1)),
    "ɛcc": df.iloc[:, 5].values.reshape((-1, 1))
}

# Define prediction function
def predict():
    selected_model = model_var.get()
    selected_target = target_var.get()
    model = models[selected_model]
    target = Targets[selected_target]

    # Get input features
    features = [float(entry.get()) for entry in feature_entries]

    # Reshape features for prediction
    features = [features]

    # Perform prediction
    Xtr, Xte, ytr, yte = train_test_split(X, target, train_size=0.7, random_state=0)
    model.fit(Xtr, ytr)
    prediction = model.predict(features)
    rounded_prediction = round(prediction[0], 4)
    prediction_label.config(text=f"Prediction: {rounded_prediction}")


# Create feature labels and entry fields
feature_labels = ["f'c (Mpa)", "tf (mm)*Ef (Mpa)", "2a (mm)", "2b (mm)"]
feature_entries = []

for i in range(len(feature_labels)):
    label = tk.Label(window, text=feature_labels[i], font=("Arial", 10, "bold"))
    label.grid(row=i, column=0, padx=5, pady=5)

    entry = tk.Entry(window)
    entry.grid(row=i, column=1, padx=5, pady=5)
    feature_entries.append(entry)

# Model selection
model_label = tk.Label(window, text="Select Model:", font=("Arial", 10, "bold"))
model_label.grid(row=len(feature_labels), column=0, padx=5, pady=5, columnspan=1)

model_var = tk.StringVar()
model_combobox = ttk.Combobox(window, textvariable=model_var, state="readonly", font=("Arial", 10))
model_combobox.grid(row=len(feature_labels), column=1, padx=5, pady=5)
model_combobox["values"] = list(models.keys())

# Target selection
target_label = tk.Label(window, text="Select Target:", font=("Arial", 10, "bold"))
target_label.grid(row=len(feature_labels) + 1, column=0, padx=10, pady=10)

target_var = tk.StringVar()
for i, target_name in enumerate(Targets.keys()):
    rb = ttk.Radiobutton(window, text=target_name, variable=target_var, value=target_name)
    rb.grid(row=len(feature_labels) + 1, column=i + 1, padx=5, pady=5)

# Prediction button
predict_button = tk.Button(window, text="Predict", command=predict, font=("Arial", 10, "bold"))
predict_button.grid(row=len(feature_labels) + 2, column=0, columnspan=len(models) + 1, padx=5, pady=5)

# Prediction label
prediction_label = tk.Label(window, text="Prediction: ", font=("Arial", 10, "bold"))
prediction_label.grid(row=len(feature_labels) + 3, column=0, columnspan=len(models) + 1, padx=5, pady=5)

window.mainloop()

