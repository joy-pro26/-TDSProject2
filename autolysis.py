import os
import sys
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import stats
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
# Existing imports remain the same
import openai
import re
from difflib import SequenceMatcher
from PIL import Image
# Add additional scientific computing imports
from scipy.stats import spearmanr, kendalltau, tsem
from sklearn.preprocessing import StandardScaler
import logging
import tempfile
# Seaborn theme setup
sns.set_theme(style="whitegrid")

# API configuration remains the same
API_URL = "https://aiproxy.sanand.workers.dev/openai/"
AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")

# Enhanced error handling for API token
if not AIPROXY_TOKEN:
    print("Error: AIPROXY_TOKEN environment variable is not set.")
    sys.exit(1)

openai.api_base = API_URL
openai.api_key = AIPROXY_TOKEN
# Create output file path.
def create_output_folder(file_path):
    """
    Creates an output folder named after the input CSV file, excluding the '.csv' extension.

    Args:
        file_path (str): The input file path.

    Returns:
        str: The path to the created output folder.
    """
    folder_name = os.path.splitext(os.path.basename(file_path))[0]
    output_folder = os.path.join(os.getcwd(), folder_name)
    os.makedirs(output_folder, exist_ok=True)
    return output_folder

# Analyzes the dataset by loading it and performing a basic summary.
def analyze_dataset(file_path):
    """
    Analyzes the dataset by loading it and performing a basic summary.

    Args:
        file_path (str): Path to the CSV file containing the dataset.

    Returns:
        tuple: A DataFrame containing the dataset and a dictionary summarizing basic analysis results.
    """
    encodings = ["utf-8", "ISO-8859-1", "utf-16","utf-32" "cp1252"]  # List of fallback encodings
    for encoding in encodings:
        try:
            # Try loading the dataset with the current encoding
            data = pd.read_csv(file_path, encoding=encoding)

            # Generate summary statistics
            summary = data.describe(include="all").to_dict()

            # Calculate missing values
            missing_values = data.isnull().sum().to_dict()

            # Calculate correlation matrix (numeric columns only)
            correlation_matrix = (
                data.select_dtypes(include=["number"]).corr().to_dict()
            )

            # Compile the analysis report
            report = {
                "summary": summary,
                "missing_values": missing_values,
                "columns_info": {col: str(data[col].dtype) for col in data.columns},
                "sample_data": data.head(3).to_dict(),  # Show first 3 rows
                "correlation_matrix": correlation_matrix,
            }

            return data, report
        except UnicodeDecodeError:
            # Try the next encoding if decoding fails
            continue
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except pd.errors.EmptyDataError:
            raise ValueError(f"The file is empty or not a valid CSV: {file_path}")
        except Exception as e:
            raise RuntimeError(f"An error occurred during analysis: {e}")
    
    # Raise an error if no encodings succeed
    raise ValueError(f"Unable to decode file: {file_path} with the tried encodings {encodings}")

# Input Validation function

def validate_inputs(data, suggestions, output_folder):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("The data argument must be a Pandas DataFrame.")
    if not isinstance(suggestions, list) or not suggestions:
        raise ValueError("The suggestions argument must be a non-empty list.")
    if not os.path.exists(output_folder):
        raise FileNotFoundError(f"The output folder '{output_folder}' does not exist.")
    if not os.access(output_folder, os.W_OK):
        raise PermissionError(f"The output folder '{output_folder}' is not writable.")

# Function to generate correlation heatmap plots
def generate_correlation_heatmap(data, numeric_columns, output_folder, config=None):
    """
    Generate a heatmap for correlation matrix with enhanced readability.

    Args:
        data (DataFrame): Dataset.
        numeric_columns (DataFrame): Numeric columns of the dataset.
        output_folder (str): Path to save the heatmap.
        config (dict): Configuration for customization (optional).

    Returns:
        list: Path to the saved heatmap image.
    """
    if numeric_columns.shape[1] > 1:
        cor_image_counter = 0
        # Configurable settings
        config = config or {}
        figsize = config.get("figsize", (12, 10))
        cmap = config.get("viridis", "coolwarm")
        cor_max_images = config.get("cor_max_images", 1)
        threshold = config.get("highlight_threshold", 0.7)
        max_columns = config.get("max_columns", 6)  # Limit columns for readability
        save_path = os.path.join(output_folder, "correlation_heatmap.png")
        # Limit columns if needed
        if numeric_columns.shape[1] > max_columns:
          numeric_columns = numeric_columns.iloc[:, :max_columns]
          print(f"Limiting to first {max_columns} numeric columns for the heatmap.")
        if cor_image_counter < cor_max_images:
          # Compute correlation matrix
          corr_matrix = numeric_columns.corr()

          plt.figure(figsize=figsize)
          sns.heatmap(
              corr_matrix,
              annot=True,
              fmt=".2f",
              cmap=cmap,
              cbar_kws={'label': 'Correlation Coefficient'},
              annot_kws={"size": 8},
          )

          # Highlight strong correlations with asterisks
          for i in range(len(corr_matrix.columns)):
              for j in range(i):
                  corr_value = corr_matrix.iloc[i, j]
                  if abs(corr_value) > threshold:
                      plt.text(j + 0.5, i + 0.5, '*', color='black',
                              ha='center', va='center', fontsize=12)

          plt.title("Correlation Matrix Heatmap", fontsize=16, weight='bold')
          plt.xticks(rotation=60, fontsize=10)
          plt.yticks(fontsize=10)
          plt.tight_layout()

          plt.savefig(save_path, dpi=300)
          plt.close()
          cor_image_counter += 1
          return [save_path]
    return []

# Function to generate distribution plots
def generate_distribution_plots(data, numeric_columns, output_folder, config=None):
    """
    Generate distribution plots (histogram and optional KDE) for numeric columns.

    Args:
        data (DataFrame): The dataset to analyze.
        numeric_columns (DataFrame): Numeric columns from the dataset.
        output_folder (str): Path to save the plots.
        config (dict): Configuration options for customization (optional).

    Returns:
        list: Paths to the saved distribution plot images.
    """
    # Configurable settings
    config = config or {}
    figsize = config.get("figsize", (10, 6))
    dist_max_images = config.get("dist_max_images", 1)
    bin_count = config.get("bin_count", 30)
    kde_color = config.get("kde_color", "green")
    hist_color = config.get("hist_color", "blue")

    images = []
    dist_image_counter = 0

    for col in numeric_columns:
        clean_data = data[col].dropna()

        # Skip empty or ID-like columns
        if clean_data.empty:
            print(f"Skipping {col}: no data available.")
            continue
        if col.lower().endswith("_id") or clean_data.nunique() == len(clean_data):
            print(f"Skipping {col}: likely an identifier column.")
            continue

        if dist_image_counter < dist_max_images:
            plt.figure(figsize=figsize)

            # Histogram using Seaborn
            sns.histplot(
                clean_data,
                kde=False,
                bins=bin_count,
                color=hist_color,
                label="Histogram"
            )

            # KDE using sns.kdeplot for customization
            sns.kdeplot(
                clean_data,
                color=kde_color,
                linewidth=2,
                label="KDE"
            )

            # Add mean and median lines
            mean_val = clean_data.mean()
            median_val = clean_data.median()
            plt.axvline(mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.2f}")
            plt.axvline(median_val, color="orange", linestyle="--", label=f"Median: {median_val:.2f}")

            # Enhance the plot
            plt.legend()
            plt.title(f"Distribution of {col}", fontsize=14, weight="bold")
            plt.xlabel(col, fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            plt.tight_layout()

            # Save the plot
            hist_path = os.path.join(output_folder, f"{col}_distribution.png")
            plt.savefig(hist_path, dpi=300)
            plt.close()

            images.append(hist_path)
            dist_image_counter += 1

    return images

# Function to generate_cluster_visualization
def generate_cluster_visualization(data, numeric_columns, output_folder, config=None):
    """
    Generate a scatter plot visualization of KMeans clusters using the first two numeric columns.

    Args:
        data (DataFrame): The dataset to analyze.
        numeric_columns (DataFrame): Numeric columns from the dataset.
        output_folder (str): Path to save the cluster visualization.
        config (dict): Configuration options (e.g., number of clusters).

    Returns:
        list: Path to the saved cluster visualization image.
    """
    # Configurable settings
    config = config or {}
    n_clusters = config.get("n_clusters", 5)
    figsize = config.get("figsize", (10, 8))
    palette = config.get("palette", "Set2")
    clust_max_images = config.get("clust_max_images", 1)
    max_columns = config.get("max_columns", 3)  # Limit the number of columns used for clustering
    max_rows = config.get("max_rows", 200)  # Reduce rows to 1,000 by default
    clust_image_counter = 0
    # Check if clustering can be performed
    if numeric_columns.shape[1] > 1:

        # Subsample rows to reduce dataset size
        if len(numeric_columns) > max_rows:
            subsampled_data = numeric_columns.sample(n=max_rows, random_state=42)
            print(f"Reduced dataset to {max_rows} rows for clustering.")
        else:
            subsampled_data = numeric_columns

        # Limit the numeric columns if necessary
        subsampled_data = subsampled_data.iloc[:, :max_columns]
        print(f"Using the first {max_columns} numeric columns for clustering.")

        # Handle missing or infinite values
        if subsampled_data.isnull().values.any():
            print("Warning: Numeric columns contain NaN values. Filling with 0.")
        if np.isinf(subsampled_data.values).any():
            print("Warning: Numeric columns contain infinite values. Replacing with 0.")
        subsampled_data = subsampled_data.fillna(0).replace([np.inf, -np.inf], 0)
            
        try:
          if clust_image_counter < clust_max_images:
              # Standardize the numeric data
              scaler = StandardScaler()
              scaled_data = scaler.fit_transform(subsampled_data)

              # Apply PCA if more than 2 columns
              if numeric_columns.shape[1] > 2:
                  pca = PCA(n_components=2)
                  scaled_data = pca.fit_transform(scaled_data)
                  print("Applied PCA to reduce dimensions for visualization.")
              # Apply KMeans clustering
              kmeans = KMeans(n_clusters=n_clusters, random_state=42)
              subsampled_data['Cluster'] = kmeans.fit_predict(scaled_data)

              # Scatter plot using the first two numeric columns
              plt.figure(figsize=figsize)
              sns.scatterplot(
                  x=scaled_data[:, 0],
                  y=scaled_data[:, 1],
                  hue=subsampled_data['Cluster'],
                  palette=palette,
                  s=100,
                  edgecolor="w",
              )

              # Add cluster centroids
              centroids = kmeans.cluster_centers_
              plt.scatter(
                  centroids[:, 0], centroids[:, 1],
                  s=200, c="red", marker="X", label="Centroids"
              )

              # Enhance plot appearance
              plt.title("Cluster Visualization", fontsize=16, weight="bold")
              plt.xlabel(numeric_columns.columns[0], fontsize=12)
              plt.ylabel(numeric_columns.columns[1], fontsize=12)
              plt.legend(title="Cluster", loc="best")
              plt.tight_layout()

              # Save the plot
              cluster_path = os.path.join(output_folder, "cluster_visualization.png")
              plt.savefig(cluster_path, dpi=100)
              plt.close()
              clust_image_counter += 1
              return [cluster_path]

        except Exception as e:
            print(f"Error generating cluster visualization: {e}")
            return []
    else:
        print("Not enough numeric columns for clustering.")
        return []

def generate_pca_visualization(data, numeric_columns, output_folder, config=None):
    """
    Generate a scatter plot visualization of the first two principal components using PCA.

    Args:
        data (DataFrame): The dataset to analyze.
        numeric_columns (DataFrame): Numeric columns from the dataset.
        output_folder (str): Path to save the PCA visualization.
        config (dict): Configuration options (e.g., figure size, alpha).

    Returns:
        list: Path to the saved PCA visualization image.
    """
    # Configurable settings
    config = config or {}
    figsize = config.get("figsize", (10, 8))
    alpha = config.get("alpha", 0.7)
    point_size = config.get("point_size", 100)
    annotate = config.get("annotate", False)  # Add point labels if True
    max_rows = config.get("max_rows", 250)  # Subsample rows if necessary
    pca_max_images = config.get("pca_max_images", 1)
    pca_image_counter = 0

    if numeric_columns.shape[1] > 1:
        try:
            if pca_image_counter < pca_max_images:

                if len(numeric_columns) > max_rows:
                    numeric_columns = numeric_columns.sample(n=max_rows, random_state=42)
                    print(f"Reduced dataset to {max_rows} rows for PCA visualization.")
                # Perform PCA
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(numeric_columns.fillna(0))

                # Create a scatter plot of the first two principal components
                plt.figure(figsize=figsize)
                scatter = sns.scatterplot(
                    x=pca_result[:, 0],
                    y=pca_result[:, 1],
                    s=point_size,
                    alpha=alpha,
                    color="blue"  # Use a single color for points
                )

                # Annotate each point with its index if enabled
                if annotate:
                    for i, (x, y) in enumerate(zip(pca_result[:, 0], pca_result[:, 1])):
                        plt.text(x, y, str(i), fontsize=8, alpha=0.6)

                # Add labels and title
                plt.title("PCA Visualization", fontsize=16, weight="bold")
                plt.xlabel("Principal Component 1", fontsize=12)
                plt.ylabel("Principal Component 2", fontsize=12)

                # Save the plot
                pca_path = os.path.join(output_folder, "pca_visualization.png")
                plt.tight_layout()
                plt.savefig(pca_path, dpi=150)
                plt.close()
                pca_image_counter+= 1
                return [pca_path]

        except Exception as e:
              print(f"Error generating PCA visualization: {e}")
              return []
    else:
        print("Not enough numeric columns for PCA.")
        return []

# Function to generate_time_series_plot
def generate_time_series_plot(data, numeric_columns, output_folder, config=None):
    """
    Generate a time series plot for numeric columns grouped by date.

    Args:
        data (DataFrame): The dataset to analyze.
        numeric_columns (DataFrame): Numeric columns from the dataset.
        output_folder (str): Path to save the time series plot.
        config (dict): Configuration options for customization (optional).

    Returns:
        list: Path to the saved time series plot image.
    """
    # Configurable settings
    config = config or {}
    date_column = config.get("date_column", "date")
    time_frequency = config.get("time_frequency", "M")  # Grouping frequency (e.g., 'M' for month, 'D' for day)
    figsize = config.get("figsize", (10, 6))
    ts_max_images = config.get("ts_max_images", 1)
    line_styles = config.get("line_styles", ["-", "--", "-.", ":"])  # Line styles for variety

    ts_image_counter = 0
    # Check if the date column exists
    if date_column in data.columns:
        try:
            # Convert the date column to datetime format
            data[date_column] = pd.to_datetime(data[date_column], format='%d-%m-%y', errors='coerce')
            data = data.dropna(subset=[date_column])
            if not data.empty : #and numeric_columns.shape[1] > 0:
                # Group numeric data by the specified time frequency
                time_series_data = numeric_columns.groupby(data[date_column].dt.to_period(time_frequency)).mean()
                print('date1')
                if not time_series_data.empty and ts_image_counter < ts_max_images:
                    # Plot time series data
                    plt.figure(figsize=figsize)
                    for i, col in enumerate(numeric_columns.columns):
                        plt.plot(
                            time_series_data.index.to_timestamp(),
                            time_series_data[col],
                            label=col,
                            linestyle=line_styles[i % len(line_styles)]  # Cycle through line styles
                        )
                    # Enhance the plot
                    plt.title("Time Series Analysis", fontsize=16, weight="bold")
                    plt.xlabel("Date", fontsize=12)
                    plt.ylabel("Average Value", fontsize=12)
                    plt.legend(title="Metrics", fontsize=10)
                    plt.grid(True, linestyle="--", alpha=0.6)
                    plt.tight_layout()

                    # Save the plot
                    time_series_path = os.path.join(output_folder, "time_series_analysis.png")
                    plt.savefig(time_series_path, dpi=300)
                    plt.close()

                    ts_image_counter += 1
                    return [time_series_path]
        except Exception as e:
            print(f"Error generating time series plot: {e}")
            return []
    else:
        print(f"Date column '{date_column}' not found in the dataset.")
    return []

# Function to Generate Box Plots
def generate_box_plots(data, numeric_columns, output_folder, config=None):
    """
    Generate box plots for numeric columns in the dataset.

    Args:
        data (DataFrame): The dataset to analyze.
        numeric_columns (DataFrame): Numeric columns from the dataset.
        output_folder (str): Path to save the box plot images.
        config (dict): Configuration options for customization (optional).

    Returns:
        list: Paths to the saved box plot images.
    """
    # Configurable settings
    config = config or {}
    figsize = config.get("figsize", (10, 6))
    bp_max_images = config.get("bp_max_images", 1)
    point_color = config.get("outlier_color", "purple")
    text_fontsize = config.get("outlier_fontsize", 8)

    images = []
    bp_image_counter = 0

    for col in numeric_columns:
        # Limit the number of box plots generated
        if bp_image_counter >= bp_max_images:
            break

        # Skip empty or non-numeric columns
        if data[col].dropna().empty:
            print(f"Skipping {col}: no valid data for box plot.")
            continue

        plt.figure(figsize=figsize)

        # Create the box plot
        sns.boxplot(
            x=data[col],
            color="skyblue",
            flierprops=dict(marker='o', color=point_color, markersize=6)
        )

        # Add vertical lines for mean and median
        mean_val = data[col].mean()
        median_val = data[col].median()
        plt.axvline(mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.2f}")
        plt.axvline(median_val, color="blue", linestyle="--", label=f"Median: {median_val:.2f}")
        plt.legend()

        # Highlight outliers
        iqr = data[col].quantile(0.75) - data[col].quantile(0.25)
        lower_bound = data[col].quantile(0.25) - 1.5 * iqr
        upper_bound = data[col].quantile(0.75) + 1.5 * iqr
        outliers = data[col][(data[col] < lower_bound) | (data[col] > upper_bound)].dropna()

        for outlier in outliers:
            plt.text(outlier, 0, str(round(outlier, 2)), color=point_color, fontsize=text_fontsize, ha="center")

        # Add title and labels
        plt.title(f"Box Plot of {col}", fontsize=16, weight="bold")
        plt.xlabel(f"Values of {col}", fontsize=12)

        # Save the plot
        box_plot_path = os.path.join(output_folder, f"{col}_box_plot.png")
        plt.tight_layout()
        plt.savefig(box_plot_path, dpi=300)
        plt.close()

        # Track generated plots
        bp_image_counter += 1
        images.append(box_plot_path)

    return images

# Function to Generate Visualisation and Graps
def generate_visualizations(data, suggestions, output_folder, config=None):
    """
    Orchestrates the generation of visualizations based on suggestions.

    Args:
        data (DataFrame): The dataset to analyze.
        suggestions (list): List of suggested analysis types.
        output_folder (str): Path to save visualizations.
        config (dict, optional): Additional configuration for visualizations.

    Returns:
        list: Paths to the generated visualization files.
    """
    validate_inputs(data, suggestions, output_folder)

    images = []
    numeric_columns = data.select_dtypes(include=np.number)

    if numeric_columns.empty:
        raise ValueError("The dataset does not contain any numeric columns.")

    # Load configuration with defaults
    config = config or {}
    max_images = config.get("max_images", 6)
    image_counter = 0

    # Suggestion-to-function mapping
    visualization_functions = {
        r"correlation": generate_correlation_heatmap,
        r"distribution": generate_distribution_plots,
        r"cluster": generate_cluster_visualization,
        r"pca": generate_pca_visualization,
        r"time series": generate_time_series_plot,
        r"box plots": generate_box_plots,
    }

    for suggestion in suggestions:
        if image_counter >= max_images:
            break
        suggestion = suggestion.lower()
        for pattern, func in visualization_functions.items():
            if re.search(pattern, suggestion, re.IGNORECASE):
                try:
                    images.extend(func(data, numeric_columns, output_folder))
                    image_counter += 1
                except Exception as e:
                    logging.error(f"Error generating visualization for {suggestion}: {e}")
                break

    return images


# Function to Generate advanced statistics
def advanced_statistical_analysis(data):
    """
    Perform advanced statistical analyses beyond basic descriptive statistics.
    Including Spearman and Kendall
    Args:
        data (pd.DataFrame): Input dataset
    
    Returns:
        dict: Comprehensive statistical analysis results
    """
    results = {}
    
    # Numeric columns selection
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    # Normality Tests
    results['normality_tests'] = {}
    max_sample_size = 5000  # Limit for Shapiro-Wilk

    for col in numeric_cols:
        clean_data = data[col].dropna()
        if len(clean_data) > 3:  # Minimum requirement for normality tests
            try:
                # Use subsampling for Shapiro-Wilk if dataset is too large
                sample_size = min(len(clean_data), max_sample_size)
                sampled_data = clean_data.sample(sample_size, random_state=42) if len(clean_data) > max_sample_size else clean_data
                shapiro_result = stats.shapiro(sampled_data)
                results['normality_tests'][col] = {
                    'statistic': shapiro_result.statistic,
                    'p_value': shapiro_result.pvalue,
                    'is_normally_distributed': shapiro_result.pvalue > 0.05,
                    'note': 'Subsampling used' if len(clean_data) > max_sample_size else 'Full dataset used'
                }
            except Exception as e:
                results['normality_tests'][col] = {'error': str(e)}
    
    # Advanced Outlier Detection
    def detect_outliers(series):
        if series.empty:
            return {}
        
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        outliers_iqr = series[
            (series < (Q1 - 1.5 * IQR)) | 
            (series > (Q3 + 1.5 * IQR))
        ]
        
        z_scores = np.abs(stats.zscore(series))
        outliers_zscore = series[z_scores > 3]
        
        return {
            'iqr_outliers_count': len(outliers_iqr),
            'zscore_outliers_count': len(outliers_zscore),
            'total_outliers': len(set(outliers_iqr.index) | set(outliers_zscore.index))
        }
    
    results['outlier_analysis'] = {
        col: detect_outliers(data[col]) for col in numeric_cols
    }
    
    # Advanced Correlation Analysis
    if len(numeric_cols) > 1:
        results['advanced_correlations'] = {
            'spearman_correlation': data[numeric_cols].corr(method='spearman').to_dict(),
            'kendall_correlation': pd.DataFrame({
                col1: {
                    col2: kendalltau(*data[[col1, col2]].dropna().T.values)[0]
                    if data[[col1, col2]].dropna().shape[0] > 0 else None
                    for col2 in numeric_cols
                } for col1 in numeric_cols
            }).to_dict()
        }

    return results

# Function to compress data for llm prompt
def compress_data_for_llm(data, max_rows=3):
    """
    Compress dataset for token-efficient LLM processing by sampling rows 
    and prioritizing numeric and categorical columns.

    Args:
        data (pd.DataFrame): Input dataset.
        max_rows (int): Maximum number of rows to sample.

    Returns:
        tuple: Compressed data (pd.DataFrame) and a lightweight summary (dict).
    """
    if data.empty:
        raise ValueError("Input data is empty.")
    
    # Select numeric and categorical columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    priority_cols = list(numeric_cols) + list(categorical_cols)

    if not priority_cols:
        raise ValueError("No numeric or categorical columns found in the dataset.")
    
    # Sample rows dynamically based on dataset size
    sample_size = min(len(data), max_rows)
    compressed_data = data[priority_cols].sample(sample_size, random_state=42)

    # Compute lightweight summary
    summary = {}
    for col in compressed_data.columns:
        if col in numeric_cols:
            summary[col] = {
                'mean': compressed_data[col].mean(),
                'std': compressed_data[col].std(),
                'min': compressed_data[col].min(),
                'max': compressed_data[col].max()
            }
        elif col in categorical_cols:
            summary[col] = {
                'unique_count': compressed_data[col].nunique(),
                'most_common': compressed_data[col].mode().iloc[0] if not compressed_data[col].mode().empty else None,
                'most_common_count': compressed_data[col].value_counts().iloc[0] if not compressed_data[col].value_counts().empty else None
            }
    
    return compressed_data, summary

# Function to create enhance prompt
def create_enhanced_prompt(report, context_level='detailed', custom_objectives=None):
    """
    Create a structured, context-rich prompt for LLM analysis.

    Args:
        report (dict): Dataset analysis report.
        context_level (str): Depth of context in prompt ('basic', 'detailed', 'expert').
        custom_objectives (list): Custom analysis objectives to include in the prompt.

    Returns:
        str: Formatted prompt for the LLM.
    """
    prompt_templates = {
        'basic': """
        Analyze the following dataset with a focus on key insights.
        Dataset Characteristics:
        - Columns: {columns}
        - Sample Data Overview: {sample_data}
        """,
        'detailed': """
        Comprehensive Data Analysis Request:
        
        1. Dataset Overview:
        - Columns: {columns}
        - Data Types: {column_types}
        - Missing Values: {missing_values}
        
        2. Statistical Context:
        - Sample Data: {sample_data}
        - Summary Statistics: {summary_stats}
        
        3. Analysis Objectives:
        {objectives}
        
        4. Deliverable:
        Provide clear and concise recommendations with an emphasis on business or research implications.
        """,
        'expert': """
        Advanced Data Exploration Protocol:
        
        Dataset Metadata:
        {comprehensive_metadata}
        
        Analysis Directives:
        - Perform advanced multi-dimensional analysis.
        - Identify hidden trends and relationships.
        - Assess data quality and biases.
        - Suggest predictive models and hypotheses.
        - Uncover non-obvious relationships

        
        Deliverable:
        - Actionable insights.
        - Business/research strategies.
        - Confidence levels for the findings.
        - Specify statistical analyses or visualizations e,g pcs, time series etc.
        """
    }

    # Default objectives if none are provided
    if custom_objectives is None:
        custom_objectives = [
            "Identify primary patterns and trends",
            "Highlight potential correlations",
            "Suggest actionable insights"
        ]

    objectives_text = "\n".join([f"- {obj}" for obj in custom_objectives])
    prompt_template = prompt_templates.get(context_level, prompt_templates['basic'])

    # Format the selected template with dataset details
    formatted_prompt = prompt_template.format(
        columns=list(report.get('columns_info', {}).keys()),
        column_types=report.get('columns_info', {}),
        missing_values=report.get('missing_values', {}),
        sample_data=report.get('sample_data', {}),
        summary_stats=report.get('summary', {}),
        comprehensive_metadata=str(report),
        objectives=objectives_text
    )

    return formatted_prompt

# Function to make llm analysis
def get_llm_analysis_suggestions(report):
    """
    Fetches LLM analysis suggestions with robust error handling, retry logic, and fallback options.

    Args:
        report (dict): Dataset analysis report.
    
    Returns:
        list: Suggested analysis types or fallback options in case of an error.
    """
    AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
    if not AIPROXY_TOKEN:
        print("Error: AIPROXY_TOKEN environment variable is not set.")
        sys.exit(1)

    api_base = "https://aiproxy.sanand.workers.dev/openai/v1"
    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
    }

    try:
        # Compress and summarize the data
        _, compressed_summary = compress_data_for_llm(pd.DataFrame(report.get('sample_data', {})))
        enhanced_prompt = create_enhanced_prompt(report, 'detailed')

        # Prepare the API payload
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": (
                    "You are a highly efficient and precise data analyst. Your role is to suggest "
                     "Based on the structure of this dataset, indicate uniquestatistical and machine learning analyses should I perform to uncover patterns and relationships?"
                    "1: What types of visualizations or analyses would best summarize the key insights from this dataset?"
                    "2: Which analyses should I prioritize to explore relationships between numeric variables in this dataset?"
                    "3: What methods can I use to identify trends, clusters, or groupings within this dataset?"
                    "4: What Exploratory Data Analysis (EDA) is to be done in this dataset?"
                    "6: What Feature Relationships can be done in this dataset?"
                    "7: What  Trend and Grouping Identification can be done in this dataset?"
                    "8: What  Visualization Techniques can be used in this dataset?"
                    "Keep your responses concise, practical, and prioritized for maximum impact such as Correlation Analysis, Distribution Analysis, Clustering Analysis, PCA Visualization."                      
                    )
                },
                {"role": "user", "content": enhanced_prompt}
            ]
        }

        # Make the API call
        response = make_api_request_with_retry(f"{api_base}/chat/completions", headers, payload)

        # Parse and return LLM suggestions
        return extract_keywords_from_response(response)

    except requests.exceptions.RequestException as e:
        print(f"HTTP error: {e}")
    except ValueError as e:
        print(f"Response parsing error: {e}")

    # Fallback suggestions
    fallback_suggestions = [
        "Correlation Analysis",
        "Distribution Analysis",
        "Clustering Analysis",
        "PCA Visualization",
        "Trend Analysis",
        "Missing Value Imputation"
    ]
    print("Returning fallback suggestions due to an error.")
    return fallback_suggestions

# Function Parses the LLM API response to extract analysis suggestions
def parse_llm_response1(response):
    """
    Parses the LLM API response to extract analysis suggestions.
    Handles both raw response objects and already parsed dictionaries.

    Args:
        response (requests.Response | dict): The API response object or a parsed JSON dictionary.

    Returns:
        list: Extracted suggestions.

    Raises:
        ValueError: If the response structure is invalid or incomplete.
    """
    try:
        # Check if the response is a raw requests.Response object
        print(response)
        if isinstance(response, requests.Response):
            response = response.json()  # Parse JSON if it's a raw response

        # Extract choices from the response dictionary
        choices = response.get("choices", [])
        if not choices:
            raise ValueError("No 'choices' found in the response.")

        # Extract content from the first choice
        message_content = choices[0].get("message", {}).get("content", "").strip()
        if not message_content:
            raise ValueError("Empty 'content' field in the response message.")

        # Split content into individual suggestions
        return [line.strip() for line in message_content.split("\n") if line.strip()]
    except (KeyError, ValueError, AttributeError) as e:
        raise ValueError(f"Error parsing LLM response: {e}")

# Function cretae key analysis to be performed

def extract_keywords_from_response(response, similarity_threshold=0.8):
    """
    Extracts and deduplicates keywords or phrases from an LLM API response, 
    ensuring the response is properly formatted.

    Args:
        response (requests.Response | dict | str | list): The API response object, parsed JSON dictionary, or textual response.
        similarity_threshold (float): Minimum similarity ratio to consider two keywords as duplicates.

    Returns:
        list: Extracted and deduplicated keywords or key phrases.

    Raises:
        ValueError: If the response is invalid or unsupported.
    """
    try:
        # Ensure response is processed into a string or list of strings
        if isinstance(response, requests.Response):
            response = response.json()  # Parse raw API response

        if isinstance(response, dict):
            # Extract content from choices if it's a parsed dictionary
            choices = response.get("choices", [])
            if not choices:
                raise ValueError("No 'choices' found in the response.")
            response_text = " ".join(choice.get("message", {}).get("content", "") for choice in choices)
        elif isinstance(response, list):
            # Join list of strings into a single string
            response_text = " ".join(response)
        elif isinstance(response, str):
            # Use the string as is
            response_text = response
        else:
            raise TypeError("Unsupported response type. Expected string, list, or dictionary.")

        # Define regex patterns for keyword extraction
        keyword_patterns = [
            r"\*\*(.*?)\*\*",            # Matches bold text (e.g., **Box Plots**)
            r"[A-Z][a-z]+(?: [A-Z][a-z]+)*",  # Matches Proper Nouns or CamelCase (e.g., Box Plots)
        ]
        combined_pattern = "|".join(keyword_patterns)

        # Extract raw matches
        matches = re.findall(combined_pattern, response_text)

        # Deduplicate matches using a similarity threshold
        deduplicated_keywords = []
        for match in matches:
            match = match.strip()
            if not match:
                continue

            # Check similarity with existing keywords
            is_duplicate = any(
                SequenceMatcher(None, match, existing).ratio() > similarity_threshold
                for existing in deduplicated_keywords
            )

            # Add to deduplicated list only if not a duplicate
            if not is_duplicate:
                deduplicated_keywords.append(match)

        return deduplicated_keywords
    except (KeyError, ValueError, TypeError) as e:
        raise ValueError(f"Error processing response: {e}")
 
 
# Function to retry api request
def make_api_request_with_retry(url, headers, payload, retries=3, backoff_factor=1, status_forcelist=None):
    """
    Makes an API request with retry logic for resilience.
    
    Args:
        url (str): API endpoint URL.
        headers (dict): HTTP headers for the request.
        payload (dict): The JSON payload to send in the request.
        retries (int): Number of retry attempts for failed requests. Default is 3.
        backoff_factor (float): Backoff factor for retry delays. Default is 1.
        status_forcelist (list): HTTP status codes that trigger a retry. Default is [500, 502, 503, 504].
    
    Returns:
        dict: Parsed JSON response from the API.
    
    Raises:
        RuntimeError: If the request fails after the specified number of retries.
    """
    if status_forcelist is None:
        status_forcelist = [500, 502, 503, 504]
    
    # Configure session with retry logic
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["POST"]  # Retry only POST requests
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount('https://', adapter)

    try:
        response = session.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()  # Return parsed JSON response
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"API request failed after {retries} retries: {e}")


# Function to create 512x512 and compress

def process_images(image_paths, output_folder,max_file_size_kb=40):
    """
    Resize images to 512x512 and compress them to the output folder.

    Args:
        image_paths (list): List of paths to input images.
        output_folder (str): Path to the folder where processed images will be saved.

    Returns:
        list: Paths to the processed images.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Ensure the output folder exists

    processed_images = []

    for img_path in image_paths:
        try:
            # Open the image
            with Image.open(img_path) as img:
                img_format = img.format  # Preserve original format
                img_resized = img.resize((512, 512), Image.Resampling.LANCZOS)  # Resize to 512x512
                #img_resized = img.resize((256, 256), Image.Resampling.LANCZOS)  # Resize to 512x512
                # Prepare the output path
                compressed_path = os.path.join(output_folder, os.path.basename(img_path))
                    # Save for other formats with compression
                img_resized.save(compressed_path, format=img_format, optimize=True, quality=50)
                # Append the processed image path to the list
                processed_images.append(compressed_path)

        except FileNotFoundError:
            print(f"File not found: '{img_path}'")
        except OSError as e:
            print(f"OS error while processing '{img_path}': {e}")
        except Exception as e:
            print(f"Unexpected error processing image '{img_path}': {e}")

    return processed_images


def prepare_structured_prompt(report, images, data, advanced_stats):
    """
    Prepare a structured and concise prompt for the LLM.
    """
    columns_info = report.get('columns_info', {})
    missing_values = report.get('missing_values', {})
    summary_stats = report.get('summary', {})
    visualization_descriptions = '\n'.join(
        [f"- **Image {i+1}:** {os.path.basename(img)}" for i, img in enumerate(set(images))]
    ) if images else 'None'

    # Simplify advanced statistics
    correlations = "Overall scores strongly correlate with quality (Spearman: 0.82). Repeatability shows a weaker correlation (Spearman: 0.49)."

    return f"""
            You are a data analyst tasked with generating a concise and actionable narrative report.
            Your report should include the following elements:
            Your report should include an excellent executive summary of insights and a cohesive story based on this analysis.
            Your report should include insights and a cohesive story based on this analysis,clearly integrating each visualization into the narrative
            Your report should Provide insights on trends, outliers, comparisons, and their implications, interpret the patterns, potential causes, and their implications for strategic planning.
            Your report should Explain insights in a narrative style, making it engaging and interesting.
            Your report shouldHighlight Which relationships stand out, and what actionable insights can be derived for decision-making
            ### Dataset Overview
            - **Columns and Data Types:** {list(columns_info.keys())}
            - **Missing Values:** {missing_values}

            ### Key Insights & Trends
            - Summary: The dataset contains {len(columns_info)} columns, with missing data in 'date' and 'by'. Numeric columns like 'overall' and 'quality' have average scores around 3.2.
            - Correlations: {correlations}

            ### Comparative Visualizations and Insights
            {visualization_descriptions}

            ### Recommendations
            Suggest actionable insights based on the findings and visualizations.

            ### Implications for Strategic Planning
            Suggest factors to be considered for strategic planning.

            ### Actionable Insights for Decision Making
            Suggest insights for decision making.
            ### Conclusion
            Summarize the key takeaways and their implications for stakeholders.
            """


def generate_insights_with_llm1(report, images, data,advanced_stats):
    """
    Generates a structured narrative using an LLM based on the dataset analysis and visualizations.

    Args:
        report (dict): Analysis results.
        images (list): Paths to generated visualizations.
        data (pd.DataFrame): Original dataset.

    Returns:
        str: Structured narrative from the LLM.
    """
    # Retrieve the API token
    AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
    if not AIPROXY_TOKEN:
        print("Error: AIPROXY_TOKEN environment variable is not set.")
        sys.exit(1)

    # API configuration
    api_base = "https://aiproxy.sanand.workers.dev/openai/v1"
    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
    }

    # Prepare analysis details for the prompt
    columns_info = report.get('columns_info', {})
    missing_values = report.get('missing_values', {})
    summary_stats = report.get('summary', {})
    advanced_stats = report.get('advanced_stats', {})

    # Describe visualizations
    visualization_descriptions = '\n'.join([
        f"- **Image {i+1}:** {os.path.basename(img)}"
        for i, img in enumerate(images)
    ]) if images else 'None'

    # Create a structured prompt

    analysis_request=prepare_structured_prompt(report, images, data, advanced_stats)

    try:
        # Prepare the payload
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are an expert data analyst and storyteller."},
                {"role": "user", "content": analysis_request}
            ]
        }
        # Make the API call
        response = requests.post(
            f"{api_base}/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()

        # Extract the narrative
        result = response.json()
        story = result["choices"][0]["message"]["content"].strip()
        return story

    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"Unexpected response structure: {e}")
        sys.exit(1)


def generate_summary_report(data):
    """
    Generates a summary report focusing on summary statistics and advanced statistics.

    Parameters:
    - data: A pandas DataFrame to summarize.

    Returns:
    - A dictionary containing summary and advanced statistics.
    """
    sumreport = {}

    # Summary Statistics
    numeric_data = data.select_dtypes(include='number')
    if not numeric_data.empty:
        sumreport['summary_statistics'] = numeric_data.describe().round(3).to_dict()

    # Advanced Statistics
    if numeric_data.shape[1] > 1:
        sumreport['advanced_statistics'] = {
            "correlation_matrix": numeric_data.corr().round(3).to_dict(),
            "covariance_matrix": numeric_data.cov().round(3).to_dict()
        }

    return sumreport

def preprocess_and_encode_png(image_path, max_size=(350, 350)):
    """
    Resize a PNG image, optimize it, and encode it in Base64.
    
    Args:
        image_path (str): Path to the input PNG image.
        max_size (tuple): Maximum width and height for resizing.
        
    Returns:
        str: Base64-encoded string of the resized, optimized PNG image.
    """
    try:
        # Open the image
        with Image.open(image_path) as img:
            # Convert to RGB if the image has an alpha channel
            img = img.convert("RGB")

            # Resize while maintaining aspect ratio
            img.thumbnail(max_size, Image.Resampling.LANCZOS)

            # Save to a temporary buffer in PNG format
            from io import BytesIO
            buffer = BytesIO()
            img.save(buffer, format="PNG", optimize=True)  # Optimize PNG
            buffer.seek(0)

            # Encode the image in Base64
            image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
            return image_base64

    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def analyze_visualization(image_path):
    try:
        # Configure logging
        model="gpt-4o-mini"
        api_base="https://aiproxy.sanand.workers.dev/openai/v1"
        logging.basicConfig(level=logging.INFO)

        # Check for the API token
        token = os.environ.get("AIPROXY_TOKEN")
        if not token:
            raise ValueError("AIPROXY_TOKEN environment variable is not set.")

        # Validate image file
        if not os.path.isfile(image_path):
            raise ValueError(f"The file {image_path} does not exist or is not a valid file.")

        encoded_image = preprocess_and_encode_png(image_path)

        # Prepare the payload
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": (
                    "You are an expert data analyst. Describe this low-resolution visualization "
                    "with a focus on key insights, patterns, and anomalies. Your response must "
                    "be concise and fit within an 85-token budget."
                )},
                {"role": "user", "content": f"The visualization is encoded in base64 below:\n{encoded_image}"}
            ]
        }

        # API headers
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        # Make the API call
        logging.info("Sending request to the API...")
        response = requests.post(f"{api_base}/chat/completions", headers=headers, json=payload, timeout=30)
        
        # Log response details
        logging.info(f"Response Status Code: {response.status_code}")
        if response.status_code != 200:
            logging.error(f"Error Response Content: {response.text}")

        # Check for HTTP errors
        response.raise_for_status()

        # Extract insights from the response
        result = response.json()
        insight = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

        # Verify token budget
        if len(insight.split()) > 85:
            logging.warning("The response exceeds the 85-token limit.")
            return "Error: Response exceeded the token limit."

        return insight

    except ValueError as e:
        logging.error(f"Configuration error: {e}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error making API request: {e}")
    except KeyError as e:
        logging.error(f"Unexpected response structure: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

    return "Error: Unable to process the request."


def process_visualizations_with_llm(images, report, data, advanced_stats):
    """
    Processes a list of visualizations using an LLM and returns insights.

    Args:
        images (list): List of image paths.
        report (dict): Analysis report context.
        data (pd.DataFrame): Original dataset.
        advanced_stats (dict): Advanced statistics.

    Returns:
        dict: A dictionary mapping image paths to LLM-generated insights.
    """
    insights = {}
    max_count = 2
    max_gen = 0
    for image_path in images:
        try:
            if  max_gen >= max_count:
                break  # Stop generating images once the limit is reached
              # Process each image individually
            else : 
                insight = analyze_visualization(image_path)
                insights[image_path] = insight
                max_gen= +1
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            insights[image_path] = "No insight generated due to an error."

    return insights

        
def generate_readme(story, images,output_folder,report,visualization_insights):
    """
    Generates a README.md file summarizing the dataset analysis, insights, and visualizations.

    Args:
        story (str): The story or analysis report generated by the LLM.
        images (list): A list of file paths to the visualizations.
        output_folder (str): Path to the output folder.
    """
    readme_path = os.path.join(output_folder, "README.md")
    with open(readme_path, "w") as f:
            
        f.write("# Summary Insight Report\n\n")
        f.write(story + "\n\n")
        f.write("## Visualization Insights\n")
        #print(f"Visualization Insights: {visualization_insights}")

        if isinstance(visualization_insights, dict):
            for img, insight in visualization_insights.items():  # Correctly unpack dictionary
                img_name = os.path.basename(img)
                f.write(f"### {img_name}\n")
                f.write(f"{insight}\n\n")
        else:
                f.write("No visualization insights available.\n\n")
        f.write("## Appendix \n")    
        if 'summary_statistics' in report:
            summary_df = pd.DataFrame(report['summary_statistics']).round(3)
            f.write(summary_df.transpose().to_markdown(index=True) + "\n\n")
        # Advanced Statistics
        f.write("## Advanced Statistics\n")
        if 'advanced_statistics' in report:
            if 'correlation_matrix' in report['advanced_statistics']:
                f.write("### Correlation Matrix\n")
                correlation_df = pd.DataFrame(report['advanced_statistics']['correlation_matrix']).round(3)
                f.write(correlation_df.to_markdown() + "\n\n")
            if 'covariance_matrix' in report['advanced_statistics']:
                f.write("### Covariance Matrix\n")
                covariance_df = pd.DataFrame(report['advanced_statistics']['covariance_matrix']).round(3)
                f.write(covariance_df.to_markdown() + "\n\n")
     
def main():
    """
    Enhanced main function to incorporate advanced analysis techniques.
    """
     # Load user configuration
    user_config = {
        "max_images": 4,  # User-specific overrides
        "max_rows": 3,
    }
    config = user_config

    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)

    # Create output folder
    output_folder = create_output_folder(file_path)
    print("Analyzing dataset...")
    # Load and analyze dataset
    data, report = analyze_dataset(file_path)
    
    sumreport=generate_summary_report(data)
    # Perform advanced statistical analysis
    advanced_stats = advanced_statistical_analysis(data)
    
    # Update report with advanced statistics
    report['advanced_stats'] = advanced_stats
    print("Asking LLM for suggestions...")
    # Get analysis suggestions
    suggestions = get_llm_analysis_suggestions(report)
   
    print("Performing suggested analyses...")
    # Perform analyses and generate visualizations
    images = generate_visualizations(data, suggestions, output_folder,config)

    # Process images for reduced size and detail
    compressed_images = process_images(images,output_folder)
    print("Generating insights using LLM...")
    visualization_insights = process_visualizations_with_llm(compressed_images,report, data,advanced_stats)
    
    # Generate insights
    print("Generating insights1 using LLM...")
    story = generate_insights_with_llm1(report, images, data,advanced_stats)
    
    print("Creating README.md...")
    # Create README
    generate_readme(story, images, output_folder,sumreport,visualization_insights)

    print(f"Analysis complete. Outputs saved in {output_folder}.")

if __name__ == "__main__":
    main()
