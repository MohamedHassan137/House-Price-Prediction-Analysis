import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error

def load_data():
    file_path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV files", "*.csv")])
    if not file_path:
        return
    global data
    data = pd.read_csv(file_path)
    clean_data()
    messagebox.showinfo("Success", "Dataset Loaded and Cleaned Successfully!")

def clean_data():
    global data
    
    # Handling missing values
    if data.isnull().sum().sum() > 0:
        data.fillna(data.median(), inplace=True)
    
    # Removing outliers using IQR
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
    
    # Convert non-numeric values if any
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

def run_analysis():
    if data is None:
        messagebox.showerror("Error", "Please load a dataset first!")
        return
    
    target = "Median_House_Value"
    features = [
        "Median_Income", "Median_Age", "Tot_Rooms", "Tot_Bedrooms",
        "Population", "Households", "Latitude", "Longitude", "Distance_to_coast",
        "Distance_to_LA", "Distance_to_SanDiego", "Distance_to_SanJose", "Distance_to_SanFrancisco"
    ]
    
    if not set(features).issubset(data.columns):
        messagebox.showerror("Error", "Some required features are missing from the dataset!")
        return
    
    X = data[features]
    y = data[target]
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True)
    
    models = {
        "Linear Regression": LinearRegression(),
        "Lasso Regression": Lasso(alpha=1.0),
        "Ridge Regression": Ridge(alpha=1.0)
    }
    
    results = {}
    plt.figure(figsize=(12, 4))
    colors = ["blue", "red", "green"]
    
    for i, (name, model) in enumerate(models.items()):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        results[name] = {"MSE": round(mse, 2), "MAE": round(mae, 2)}
        
        # Plot actual vs predicted values
        plt.subplot(1, 3, i + 1)
        plt.scatter(y_test, y_pred, color=colors[i], alpha=0.5)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(f"{name}")
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', linestyle='dashed')
    
    plt.tight_layout()
    plt.show()
    update_table(results)

def update_table(results):
    for row in table.get_children():
        table.delete(row)
    for model, metrics in results.items():
        table.insert("", "end", values=(model, metrics["MSE"], metrics["MAE"]))

# GUI Setup
root = tk.Tk()
root.title("House Price Prediction Analysis")
root.geometry("500x400")

frame = tk.Frame(root)
frame.pack(pady=20)

load_button = tk.Button(frame, text="Load Dataset", command=load_data)
load_button.grid(row=0, column=0, padx=10)

analyze_button = tk.Button(frame, text="Run Analysis", command=run_analysis)
analyze_button.grid(row=0, column=1, padx=10)

columns = ("Model", "MSE", "MAE")
table = ttk.Treeview(root, columns=columns, show="headings")
for col in columns:
    table.heading(col, text=col)
    table.column(col, width=150)
table.pack(pady=20)

root.mainloop()
