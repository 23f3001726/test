# /// script
# requires-python = ">=3.10.12"
# dependencies = [
#     "chardet",
#     "jupyter",
#     "matplotlib",
#     "numpy",
#     "opencv-python",
#     "pandas",
#     "plotly",
#     "requests",
#     "scikit-learn",
#     "scipy",
#     "seaborn",
#     "statsmodels",
#     "torch",
#     "scikit-learn",
#     "statsmodels",
#     "networkx"
# ]
# ///


#Importing
import os
import sys
import requests
import json
import base64
import chardet
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score




#Path of csv file
path = str(sys.argv[-1])

#Open API Key from env variables
api_key = os.environ["AIPROXY_TOKEN"]


#ChatGpt request api function
def chatgpt(message):
    try:
        # Make the POST request to the OpenAI API
        response = requests.post(
            "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "gpt-4o-mini",
                "messages": message
            }
        )

        # Check if the response status code is 200 (OK)
        if response.status_code == 200:
            try:
                # Parse the response JSON and return the model's message content
                return (response.json()["choices"][0]["message"]["content"])

            except Exception as e:
                print(f"Error parsing JSON response: {str(e)}")
                print(f"Response Content: {response.json()}")

        # If status code isn't 200, handle non-OK responses
        else:
            error_message = response.json().get("error", {}).get("message", "Unknown error occurred.")
            print(f"Error: {response.status_code} - {error_message}")

    except requests.exceptions.RequestException as e:
        # Catch network or request-related errors
        print(f"Request failed: {str(e)}")
    except Exception as e:
        # Catch other unexpected errors
        print(f"An unexpected error occurred: {str(e)}")

#Change image to base64
def base64_encode(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

#Get Graph Type
def graph(statement):
    message =  [
                {"role": "system", "content": "You are an person excelling in data visualisation using Seaborn. Give me the single best graph to use for following problem statement. Give only graph name, without any extra text."},
                {"role": "user", "content": statement }
            ]
    return str(chatgpt(message))

#Get code for graph generation graph
def analyse_graph(details, statement, graph, filename):
    message = [
                {"role": "system", "content": "You are an expert of data analysis, data visualisation. Give me code(No extra text) for analysis statement over given file. Code must be error-free with proper exception handling(atleat print error). Don't use deprecated/to be deprecated methods. Use " + graph + " for visualisation using seaborn(enhance with titles, axis labels, legends, colors, annotations, and enhanced customization). Choose data points judiciously, for visualisation(there might be too much data points on graph, making it cluttered and un-readable)."},
                {"role": "user", "content": "File Details are: " + details +
                 "Analysis Statement: " + statement +"Save the graph with name "+ filename + ". Make sure that graph is clutter-free and human readable."}
            ]
    return chatgpt(message)[9:][:-4]

#Get summary for graph
def summarise_graph(statement, base64_image):
    message = [
                {"role": "system", "content": "You are an expert of analysisng different types of graphs. Give me detailed analysis( in markdown format ) with possible implications for the graph for problem statement on file with without any extra text"},
                {"role": "user", "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "detail": "low",
                            "url": "data:image/png;base64," + base64_image
                        }
                    },
                    {
                        "type": "text",
                        "text": "Problem Statement: " + statement 
                    }
                ] }
            ]
    return chatgpt(message)

#get file details
def file_details(file_path):
    # Detect the encoding of the file
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']

    # Read the CSV file with the detected encoding
    df = pd.read_csv(file_path, encoding=encoding)

    # Initialize the details dictionary
    details = {
        'file_path': file_path,
        'encoding': encoding,
        'columns': {},
        'summary_statistics': df.describe(include='all').to_dict(),
        'missing_values': df.isnull().sum().to_dict()
    }

    # Add column details (name, type, example value)
    for col in df.columns:
        details['columns'][col] = {
            'type': str(df[col].dtype),
            'example': df[col].iloc[0]  # Get the first value as an example
        }

    return details

# Custom function to handle numpy types
def custom_serializer(obj):
    if isinstance(obj, np.int64):
        return int(obj)  # Convert numpy.int64 to int
    raise TypeError(f"Type {type(obj)} not serializable")

detail = file_details(path)
#Drop empty numerical rows


# File path of the CSV file
file_path = detail['file_path']

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path, encoding= detail['encoding'])

# Drop rows where any numerical column has a NaN value
df.dropna(subset=df.select_dtypes(include=['float64', 'int64']).columns, inplace=True)

# Save the updated DataFrame back to the same CSV file (overwrite original file)
df.to_csv(file_path, index=False)

detail = file_details(path) #Get updated details

#Get important numerical features

message = [
                {"role": "system", "content": "Given a list of CSV columns with names, data types, and sample values, identify the top numerical features for data analysis. Return a Python list of the most important numerical feature columns, sorted by importance, with a limit of 5 columns." },
                {"role": "user", "content": "Column Details: " + json.dumps(detail['columns'], default = custom_serializer) + f", Summary Statistics: {detail['summary_statistics']}" + "Your output should look like: ['col1', ..., 'col5']"}
            ]
# Using regular expression to find the list in the response
match = re.search(r"\[([^\]]+)\]", chatgpt(message))

# Extract the content inside the brackets and split by commas and store it in detail dictionary
detail['important_features'] = [col.strip().strip("'") for col in match.group(1).split(",")]

#Variable to Store analysis temporarily
readme = ''

#Get Basic Analysis

message = [
            {"role": "system", "content": "You are an expert of data analysis.  Give me concise analysis( in markdown format ) with possible implications for file, with without any extra text"},
            {"role": "user", "content": f"Basic Analysis:- File Path: {detail['file_path']}, Encoding: {detail['encoding']}, Columns: {json.dumps(detail['columns'],default = custom_serializer) }, Columns Summary: {json.dumps(detail['summary_statistics'], default = custom_serializer)}"  }
        ]
response = chatgpt(message)
cleaned_response = re.sub(r'```markdown\n# .+\n', '', response)  # For ```markdown case
cleaned_response = re.sub(r'```(\n| )', '', response)
readme += cleaned_response + "\n"
readme += "\n"

#Generate Correlation Heatmap

# Path to the CSV file
csv_file = detail['file_path']  # Replace with your actual CSV file path

# List of columns to analyze
column_names = detail['important_features']  # Replace with your actual column names

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file, encoding = detail['encoding'])

# Check if all specified columns are in the dataframe
missing_cols = [col for col in column_names if col not in df.columns]
if missing_cols:
    print(f"Warning: The following columns are not in the CSV file: {', '.join(missing_cols)}")
else:
    # Filter the DataFrame to only include the specified columns
    df_selected = df[column_names]

    # Convert the DataFrame to a NumPy array
    data_array = df_selected.to_numpy()

    # Compute the correlation matrix using NumPy
    corr_matrix = np.corrcoef(data_array, rowvar=False)

    #add to detail
    detail['corr_matrix'] = np.array2string(corr_matrix, formatter={'float_kind': lambda x: f"{x:.2f}"})

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, xticklabels=column_names, yticklabels=column_names)

    # Add a title
    plt.title('Correlation Heatmap')

    # Save the heatmap as a PNG file
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.savefig('heatmap.png')  # Save the heatmap image


readme+= "![Failed To Load Image](heatmap.png)\n"
response = (summarise_graph("Correlation Heatmap", base64_encode('heatmap.png')))
cleaned_response = re.sub(r'```markdown\n# .+\n', '', response)  # For ```markdown case
cleaned_response = re.sub(r'```(\n| )', '', response)
readme += cleaned_response + "\n"
readme += '\n'

# Check if all specified columns are in the dataframe
missing_cols = [col for col in column_names if col not in df.columns]
if missing_cols:
    print(f"Warning: The following columns are not in the CSV file: {', '.join(missing_cols)}")
else:
    # Filter the DataFrame to only include the specified columns
    df_selected = df[column_names]

    # Define the target variable (dependent variable)
    target = column_names[0]  # Replace with the column you want to predict

    # Define the feature variables (independent variables)
    features = [col for col in column_names if col != target]

    # Prepare the feature and target datasets
    X = df_selected[features]  # Features
    y = df_selected[target]    # Target

    # Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Linear Regression model
    model = LinearRegression()

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Predict the target variable using the test set
    y_pred = model.predict(X_test)

    # Evaluate the model's performance and store it in a variable
    reg = "Model Performance Evaluation:-" + f"R² Score: {r2_score(y_test, y_pred):.4f}," + f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}"

    # Add the regression coefficients (slopes) and intercept to variable
    reg += ("\nRegression Coefficients (Slopes):")
    for i, col in enumerate(features):
        reg += (f"{col}: {model.coef_[i]:.4f}")
    reg += (f"Intercept: {model.intercept_:.4f}")
    detail['regression'] = reg

    # Create a plot to compare predicted vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # 45-degree line
    plt.title("Predicted vs Actual Values")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    # Save the plot as 'regression.png'
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.savefig('regression.png')  # Save the plot as a PNG image

readme+= "![Failed To Load Image](regression.png)\n" #adding image
response = (summarise_graph("Regression Analysis", base64_encode('regression.png'))) #Summary
cleaned_response = re.sub(r'```markdown\n# .+\n', '', response)  # For ```markdown case
cleaned_response = re.sub(r'```(\n| )', '', response)
readme += cleaned_response + "\n"
readme += '\n'

# Identify numerical columns
numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()

# Visualizing the distribution of the numerical columns before detection
plt.figure(figsize=(12, 8))
df[numerical_columns].hist(bins=30, figsize=(12, 8), color='skyblue', edgecolor='black')
plt.suptitle('Distribution of Numerical Columns')
plt.tight_layout()
plt.savefig('distribution_before_outliers.png')  # Save before detecting outliers

# Outlier Detection

## 1. Z-Score Method (for normally distributed data)
z_scores = np.abs(stats.zscore(df[numerical_columns]))  # Z-scores for each value
z_threshold = 3  # Values with Z-score greater than 3 are considered outliers
z_outliers = (z_scores > z_threshold)

## 2. IQR Method (Interquartile Range)
Q1 = df[numerical_columns].quantile(0.25)
Q3 = df[numerical_columns].quantile(0.75)
IQR = Q3 - Q1
iqr_outliers = ((df[numerical_columns] < (Q1 - 1.5 * IQR)) | (df[numerical_columns] > (Q3 + 1.5 * IQR)))

## 3. Isolation Forest Method (Anomaly Detection)
iso_forest = IsolationForest(contamination=0.05)  # Assuming 5% contamination
iso_outliers = iso_forest.fit_predict(df[numerical_columns]) == -1  # -1 is for anomalies

# Combine the outlier detection results
outliers_combined = z_outliers | iqr_outliers | iso_outliers.reshape(-1, 1)  # Combine all methods

# Create a mask for outliers (True for outliers)
outlier_mask = outliers_combined.any(axis=1)

# Visualizing Outliers

# Create a plot highlighting the outliers
plt.figure(figsize=(12, 8))
for col in numerical_columns:
    plt.scatter(df.index, df[col], c=outlier_mask, cmap='coolwarm', label=f'{col}')

plt.title('Outliers in Numerical Data')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend(loc='upper right')
plt.tight_layout()

# Save the plot as 'outlier.png'
plt.savefig('outlier.png')  # Save the plot with outliers highlighted

# Optionally, display the plot

# Add outliers (indexes) to details
outlier_indices = df[outlier_mask].index
detail['outlier_indices'] = outlier_indices.tolist()

readme+= "![Failed To Load Image](distribution_before_outliers.png)" #adding image
readme+= "![Failed To Load Image](outlier.png)\n" #adding image
response = (summarise_graph("Outlier and Anomaly Detection", base64_encode('outlier.png'))) #adding summary
cleaned_response = re.sub(r'```markdown\n# .+\n', '', response)  # For ```markdown case
cleaned_response = re.sub(r'```(\n| )', '', response)
readme += cleaned_response + "\n"
readme += '\n'

# Select numerical columns for clustering
columns_to_cluster = df.select_dtypes(include=[np.number]).columns.tolist()

# Standardize the data (important for distance-based algorithms like K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[columns_to_cluster])

# Optional: Reduce dimensions for better visualization (using PCA for 2D projection)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Range of k values to test (2 to 10 clusters)
range_k = range(2, 11)

sil_scores = []
kmeans_labels_all = {}

for k in range_k:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)

    # Calculate the silhouette score for this k
    sil_score = silhouette_score(X_scaled, kmeans_labels)
    sil_scores.append(sil_score)
    kmeans_labels_all[k] = kmeans_labels

# Find the optimal k (highest silhouette score)
optimal_k = range_k[np.argmax(sil_scores)]
detail['clusters'] = 4 #add to file details

# K-Means clustering with the optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Plotting the final K-Means Clusters with the optimal k
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', s=50, alpha=0.7)
plt.title(f'K-Means Clustering with k={optimal_k}')
plt.xlabel('PCA Component: 1')
plt.ylabel('PCA Component: 2')
plt.colorbar(label='Cluster Label')
plt.tight_layout()
plt.savefig('clustering.png')  # Save plot

readme+= "![Failed To Load Image](clustering.png)\n" #adding image
response = (summarise_graph("Clustering", base64_encode('clustering.png'))) #adding summary
cleaned_response = re.sub(r'```markdown\n# .+\n', '', response)  # For ```markdown case
cleaned_response = re.sub(r'```(\n| )', '', response)
readme += cleaned_response + "\n"
readme += '\n'

#Getting problem statements into a list
message = [
            {"role": "system", "content": "You are an expert data analyst. Give me the best problem statement for further analysis. Just give one line statement"},
            {"role": "user", "content": f"File Details are:- Filepath: {detail['file_path']}, Encoding: {detail['encoding']}, Columns: {json.dumps(detail['columns'], default = custom_serializer)}" + 
             f"Basic Analysis:- Important Columns: {detail['important_features']}, Correlation Matrix: {detail['corr_matrix']}, Regression Analysis: {detail['regression']}" +
             ". Give me statements that give me best and useful insights."}
        ]
statement = chatgpt(message)

blob = '# ' + "Analysis Statement: " + statement + "\n" #Store problem statements

Flag = True #Flag for errors

details = f"File Path: {detail['file_path']}, Encoding: {detail['encoding']}, Columns: {json.dumps(detail['columns'],default = custom_serializer) }"
print("Statement:", statement) #log to console
Graph = graph(statement) #Get graph type
t = "statement.png" #png file path
blob += "![Failed To Load Image](" + t + ")\n" #adding image to variable
try:
  #generate graph for variable  
  exec(analyse_graph(details, statement, Graph, t))
  
except Exception as e:
  print('Graph Generation Error:', e) #log error to console

try:
  #read graph and convert it to base64_image
  base64_image = base64_encode(t)
except FileNotFoundError as e:
  print('Graph Not Found Error:', e) #log error to console
except Exception as e:
  print('Graph Encoding Error:', e) #log error to console

try: 
  #add summary to variable
  blob += summarise_graph( statement, base64_image) + "\n"
except Exception as e:
  Flag = False #set flag to false

if Flag:
  readme += blob #add to readme


#Writing to file
# Open the README.md file in write mode using a context manager
with open('README.md', 'w') as file:
    # Write the entire string (which already has newlines) to the file
    file.write(readme)
file.close()
