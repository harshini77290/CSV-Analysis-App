from flask import Flask, request, render_template, send_from_directory, jsonify
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dotenv import load_dotenv
import re

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
UPLOAD_FOLDER = 'plots'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

data = None

# Define comprehensive responses for various data-related questions
keywords_responses = {
    "mean": ("The mean, or average, is calculated by summing all values in a dataset and dividing by the number of values. "
             "It provides a central value of the data distribution and is useful for understanding the general tendency of the data."),
    
    "median": ("The median is the middle value in a dataset when ordered from smallest to largest. "
               "It is particularly useful when you have outliers or skewed data, as it represents the point where half of the data falls below and half above."),
    
    "mode": ("The mode is the value that appears most frequently in a dataset. "
             "A dataset can have one mode, more than one mode, or no mode at all. It helps identify the most common values within the data."),
    
    "standard deviation": ("Standard deviation measures the amount of variation or dispersion in a dataset. "
                           "A low standard deviation means the data points are close to the mean, while a high standard deviation indicates a wider spread."),
    
    "correlation": ("Correlation measures the strength and direction of the linear relationship between two variables. "
                    "It ranges from -1 to 1, where -1 indicates a perfect negative relationship, 1 indicates a perfect positive relationship, and 0 indicates no linear relationship."),
    
    "histogram": ("A histogram is a graphical representation of the distribution of numerical data. "
                  "It uses bars to show the frequency of data within certain ranges or bins, helping to visualize the distribution and identify patterns or outliers."),
    
    "scatter plot": ("A scatter plot displays the relationship between two numeric variables using dots. "
                     "It helps to identify correlations, trends, and potential outliers by plotting one variable on the x-axis and another on the y-axis."),
    
    "line plot": ("A line plot shows trends over time or ordered data points by connecting data points with lines. "
                  "It is useful for visualizing changes and trends in data over a continuous range."),
    
    "box plot": ("A box plot displays the distribution of a dataset through its quartiles and outliers. "
                 "It shows the median, upper and lower quartiles, and potential outliers, helping to understand the spread and skewness of the data."),
    
    "heatmap": ("A heatmap represents data values through color gradients. "
                "It is useful for visualizing complex data patterns and correlations in a matrix format, where the intensity of color represents the magnitude of values."),
    
    "pair plot": ("A pair plot visualizes relationships between multiple variables in a dataset. "
                  "It creates scatter plots for each pair of variables and histograms for individual variables, helping to identify patterns and correlations."),
    
    "residuals": ("Residuals are the differences between observed and predicted values in a regression model. "
                  "Analyzing residuals helps assess the fit of the model and identify patterns or biases that the model may not have captured."),
    
    "feature selection": ("Feature selection involves choosing the most relevant features for a model. "
                          "It improves model performance, reduces complexity, and helps to prevent overfitting by focusing on the most informative variables."),
    
    "hypothesis testing": ("Hypothesis testing is a statistical method used to determine if there is enough evidence to reject a null hypothesis. "
                           "It helps in making data-driven decisions and assessing the validity of assumptions based on sample data."),
    
    "regression analysis": ("Regression analysis explores the relationship between a dependent variable and one or more independent variables. "
                            "It helps in predicting outcomes, understanding variable relationships, and assessing the strength of associations."),
    
    "classification": ("Classification is a machine learning task where the goal is to predict categorical labels based on input features. "
                       "It involves training models to distinguish between different classes and is used in various applications such as spam detection and medical diagnosis."),
    
    "clustering": ("Clustering groups similar data points together based on their features. "
                   "It is an unsupervised learning technique used to identify natural groupings or patterns within the data, such as customer segmentation."),
    
    "PCA": ("Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms data into orthogonal components. "
            "It captures the most variance in fewer dimensions, simplifying the data while retaining essential patterns."),
    
    "time series analysis": ("Time series analysis involves analyzing data points collected at specific time intervals. "
                             "It helps in forecasting future values, identifying seasonal trends, and understanding temporal patterns in the data."),
    
    "data drift": ("Data drift refers to changes in the distribution of data over time. "
                   "Monitoring data drift is crucial for maintaining model performance and ensuring that predictions remain accurate as the data evolves."),
    
    "outlier detection": ("Outlier detection identifies data points that deviate significantly from the majority of the data. "
                          "It helps in understanding anomalies, improving data quality, and ensuring that the model is not unduly influenced by extreme values."),
    
    "normal distribution": ("A normal distribution, or Gaussian distribution, is a probability distribution with a bell-shaped curve. "
                            "It is commonly used in statistics and data analysis to represent data that clusters around a central value."),
    
    "confidence interval": ("A confidence interval is a range of values within which a parameter is expected to fall with a certain probability. "
                            "It provides an estimate of the uncertainty around a measure, giving a range where the true value is likely to lie."),
    
    "sample size": ("Sample size refers to the number of observations or data points collected in a study. "
                    "Larger sample sizes generally lead to more reliable and accurate results, reducing the impact of random variability."),
    
    "bias-variance tradeoff": ("The bias-variance tradeoff is the balance between a model’s accuracy on training data (bias) and its generalization to unseen data (variance). "
                               "Managing this tradeoff is crucial for building models that perform well on both training and test data."),
    
    "data cleaning": ("Data cleaning involves removing or correcting errors and inconsistencies in the dataset. "
                      "This process ensures that the data is accurate and reliable for analysis, improving the quality of insights derived from the data."),
    
    "data transformation": ("Data transformation refers to converting data from one format or structure into another. "
                            "This includes operations like normalization, aggregation, and encoding, which prepare the data for analysis and modeling."),
    
    "data aggregation": ("Data aggregation combines multiple data points into a summary form, such as calculating the sum or average. "
                         "It helps in simplifying and analyzing large datasets by providing a concise overview of key metrics."),
    
    "data imputation": ("Data imputation fills in missing values in a dataset using methods like mean substitution, median substitution, or predictive models. "
                        "It ensures that the dataset remains complete and usable for analysis, preventing issues that can arise from missing data.")
}

def read_csv(file_path):
    return pd.read_csv(file_path)

# Function to perform basic statistical analysis
def statistical_analysis(data):
    numeric_data = data.select_dtypes(include=[float, int])
    non_numeric_data = data.select_dtypes(exclude=[float, int])

    analysis = {
        'mean': numeric_data.mean().to_dict(),
        'median': numeric_data.median().to_dict(),
        'mode': numeric_data.mode().iloc[0].to_dict() if not numeric_data.mode().empty else None,
        'std_deviation': numeric_data.std().to_dict(),
        'correlation': numeric_data.corr().to_dict()
    }

    # Handle non-numeric columns
    non_numeric_mode = non_numeric_data.mode().iloc[0].to_dict() if not non_numeric_data.mode().empty else None
    if non_numeric_mode is not None:
        analysis['non_numeric_mode'] = non_numeric_mode

    return analysis

# Function to generate plots
def generate_plots(data, filename_prefix):
    sns.set(style="darkgrid")

    # Histograms
    hist_file = os.path.join(UPLOAD_FOLDER, f'{filename_prefix}_histogram.png')
    numeric_data = data.select_dtypes(include=[float, int])
    num_columns = len(numeric_data.columns)
    num_rows = (num_columns // 3) + (1 if num_columns % 3 != 0 else 0)
    numeric_data.hist(bins=30, figsize=(15, 5*num_rows), layout=(num_rows, 3))
    plt.suptitle('Histograms')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(hist_file)
    plt.clf()

    # Scatter plots (example using first two numeric columns)
    scatter_file = None
    if len(numeric_data.columns) >= 2:
        scatter_file = os.path.join(UPLOAD_FOLDER, f'{filename_prefix}_scatter.png')
        sns.scatterplot(data=numeric_data, x=numeric_data.columns[0], y=numeric_data.columns[1])
        plt.title('Scatter Plot')
        plt.savefig(scatter_file)
        plt.clf()

    # Line plots (example using first numeric column)
    line_file = os.path.join(UPLOAD_FOLDER, f'{filename_prefix}_line.png')
    numeric_data[numeric_data.columns[0]].plot(kind='line')
    plt.title('Line Plot')
    plt.savefig(line_file)
    plt.clf()

    return hist_file, scatter_file, line_file

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global data
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        data = read_csv(file_path)
        print(f"File uploaded successfully! File path: {file_path}")
        analysis_results = statistical_analysis(data)
        hist_file, scatter_file, line_file = generate_plots(data, file.filename)

        return render_template('index.html', analysis=analysis_results,
                               hist_file=hist_file, scatter_file=scatter_file, line_file=line_file)

@app.route('/plots/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/ask', methods=['POST'])
def ask_question():
    global data
    question = request.form['question'].lower()
    response = "Sorry, I don't have an answer for that."

    if data is not None:
        # Check for data-specific questions about columns first
        column_names = data.columns.str.lower()
        column_name_match = next((col for col in column_names if col in question), None)
        
        if column_name_match:
            column = data[column_name_match]
            if "mean" in question:
                response = f"The mean of the column '{column_name_match}' is: {column.mean()}"
            elif "median" in question:
                response = f"The median of the column '{column_name_match}' is: {column.median()}"
            elif "mode" in question:
                response = f"The mode of the column '{column_name_match}' is: {column.mode().iloc[0] if not column.mode().empty else 'No mode'}"
            elif "standard deviation" in question:
                response = f"The standard deviation of the column '{column_name_match}' is: {column.std()}"
            elif "correlation" in question:
                correlation = data.corr().get(column_name_match, 'Column not found')
                response = f"The correlation of the column '{column_name_match}' with other columns is: {correlation.to_dict()}"
        
        # Generate a response based on keywords and data if no column-specific question was found
        if response == "Sorry, I don't have an answer for that.":
            for keyword, answer in keywords_responses.items():
                if keyword in question:
                    response = answer
                    break

    return jsonify({"answer": response})

if __name__ == '__main__':
    app.run(debug=True)