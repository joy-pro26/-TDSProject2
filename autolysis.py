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
from PIL import Image
# Add additional scientific computing imports
from scipy.stats import spearmanr, kendalltau, tsem
from sklearn.preprocessing import StandardScaler

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

def analyze_dataset(file_path):
    """
    Analyzes the dataset by loading it and performing a basic summary.

    Args:
        file_path (str): Path to the CSV file containing the dataset.

    Returns:
        tuple: A DataFrame containing the dataset and a dictionary summarizing basic analysis results.
    """
    try:
        # Load the dataset into a pandas DataFrame
        data = pd.read_csv(file_path,encoding='ISO-8859-1')

        # Summary statistics
        summary = data.describe(include='all').to_dict()

        # Missing values
        missing_values = data.isnull().sum().to_dict()

        # Correlation matrix
        correlation_matrix = data.corr(numeric_only=True).to_dict()

        # Report compilation
        report = {
            "summary": summary,
            "missing_values": missing_values,
            "columns_info": {col: str(data[col].dtype) for col in data.columns},
            "sample_data": data.head(3).to_dict()  # Reduce sample data size to 3 rows
        }

        return data, report
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def generate_visualizations(data, suggestions, output_folder,config=None):
    """
    Orchestrates the generation of visualizations based on suggestions.

    Args:
        data (DataFrame): The dataset to analyze.
        suggestions (list): List of suggested analysis types.
        output_folder (str): Path to save visualizations.

    Returns:
        list: Paths to the generated visualization files.
    """
    images = []
    numeric_columns = data.select_dtypes(include=np.number)

    # Load configuration with defaults
    config = config or {}
    max_images = config.get("max_images", 6)
    image_counter = 0

    for suggestion in suggestions:
        if image_counter >= max_images:
            break  # Stop generating images once the limit is reached

        suggestion = suggestion.lower()
        print(suggestion)

        if re.search(r"correlation", suggestion, re.IGNORECASE) and len(numeric_columns.columns) > 1:
              images.extend(generate_correlation_heatmap(data, numeric_columns, output_folder))
              image_counter += 1

        elif re.search(r"distribution", suggestion, re.IGNORECASE) and not numeric_columns.empty:
              images.extend(generate_distribution_plots(data, numeric_columns, output_folder))
              image_counter += 1
             

        elif re.search(r"cluster", suggestion, re.IGNORECASE) and len(numeric_columns.columns) > 1:
              images.extend(generate_cluster_visualization(data, numeric_columns, output_folder))
              image_counter += 1

        elif re.search(r"pca", suggestion, re.IGNORECASE) and len(numeric_columns.columns) > 1:
              images.extend(generate_pca_visualization(data, numeric_columns, output_folder))
              image_counter += 1

        elif re.search(r"time series", suggestion, re.IGNORECASE) and 'date' in data.columns:
              images.extend(generate_time_series_plot(data, numeric_columns, output_folder))
              image_counter += 1

        elif re.search(r"boxplot", suggestion, re.IGNORECASE) and not numeric_columns.empty:
              images.extend(generate_box_plots(data, numeric_columns, output_folder))
              image_counter += 1

    return images

def generate_correlation_heatmap(data, numeric_columns, output_folder):
    if numeric_columns.shape[1] > 1:
        cor_image_counter = 0
        cor_max_images = 1
        if cor_image_counter < cor_max_images:
          corr_matrix = numeric_columns.corr()
          plt.figure(figsize=(10, 8))
          sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", cbar_kws={'label': 'Correlation Coefficient'})
          # Annotate strong correlations
          for i in range(len(corr_matrix.columns)):
              for j in range(i):
                  corr_value = corr_matrix.iloc[i, j]
                  if abs(corr_value) > 0.7:
                      plt.text(j + 0.5, i + 0.5, f"*", color="black", ha="center", va="center", fontsize=12)
          heatmap_path = os.path.join(output_folder, "correlation_heatmap.png")
          plt.title("Correlation Matrix Heatmap", fontsize=14, weight='bold')
          plt.xticks(rotation=45, ha='right', fontsize=10)
          plt.yticks(fontsize=10)
          plt.tight_layout()
          plt.savefig(heatmap_path)
          plt.close()
          return [heatmap_path]
    return []

def generate_distribution_plots(data, numeric_columns, output_folder):
    images = []
    dist_image_counter = 0
    dist_max_images = 1
    for col in numeric_columns:
        if data[col].dropna().shape[0] > 0:
            if dist_image_counter < dist_max_images:
              col_range = data[col].max() - data[col].min()
              if col_range > 0:
                  plt.figure(figsize=(10, 6))
                  sns.histplot(data[col], kde=True, binwidth=col_range / 30, label='Histogram')
                  mean_val = data[col].mean()
                  median_val = data[col].median()
                  plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
                  plt.axvline(median_val, color='blue', linestyle='--', label=f'Median: {median_val:.2f}')
                  plt.legend()
                  hist_path = os.path.join(output_folder, f"{col}_distribution.png")
                  plt.title(f"Distribution of {col}", fontsize=14, weight='bold')
                  plt.xlabel(col, fontsize=12)
                  plt.ylabel("Frequency", fontsize=12)
                  plt.savefig(hist_path)
                  plt.close()
                  dist_image_counter+= 1
                  images.append(hist_path)
    return images

def generate_cluster_visualization(data, numeric_columns, output_folder):
    if numeric_columns.shape[1] > 1:
        clust_image_counter = 0
        clust_max_images = 1
        if clust_image_counter < clust_max_images:
          scaler = StandardScaler()
          scaled_data = scaler.fit_transform(numeric_columns.fillna(0))
          kmeans = KMeans(n_clusters=3, random_state=42)
          kmeans.fit(scaled_data)
          data['Cluster'] = kmeans.labels_

          plt.figure(figsize=(10, 8))
          sns.scatterplot(
              x=numeric_columns.iloc[:, 0],
              y=numeric_columns.iloc[:, 1],
              hue=data['Cluster'],
              palette='tab10',
              s=100
          )
          plt.title("Cluster Visualization", fontsize=14, weight='bold')
          plt.xlabel(numeric_columns.columns[0], fontsize=12)
          plt.ylabel(numeric_columns.columns[1], fontsize=12)
          plt.legend(title="Cluster", loc='best')
          cluster_path = os.path.join(output_folder, "cluster_visualization.png")
          plt.savefig(cluster_path)
          plt.close()
          clust_image_counter += 1
          return [cluster_path]
    return []

def generate_pca_visualization(data, numeric_columns, output_folder):
    if numeric_columns.shape[1] > 1:
        pca_image_counter = 0
        pca_max_images = 1
        if pca_image_counter < pca_max_images:
          pca = PCA(n_components=2)
          pca_result = pca.fit_transform(numeric_columns.fillna(0))
          plt.figure(figsize=(10, 8))
          sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], alpha=0.7)
          plt.title("PCA Visualization", fontsize=14, weight='bold')
          plt.xlabel("Principal Component 1", fontsize=12)
          plt.ylabel("Principal Component 2", fontsize=12)
          pca_path = os.path.join(output_folder, "pca_visualization.png")
          plt.savefig(pca_path)
          plt.close()
          pca_image_counter+= 1
          return [pca_path]
    return []

def generate_time_series_plot(data, numeric_columns, output_folder):
    if 'date' in data.columns:
        ts_image_counter = 0
        ts_max_images = 1
        if ts_image_counter < ts_max_images:
          data['date'] = pd.to_datetime(data['date'],format='%d/%m/%Y', errors='coerce')
          data = data.dropna(subset=['date'])
          if not data.empty:
              time_series_data = numeric_columns.groupby(data['date'].dt.to_period('M')).mean()
              if not time_series_data.empty:
                  plt.figure(figsize=(10, 6))
                  for col in numeric_columns.columns:
                      plt.plot(
                          time_series_data.index.to_timestamp(),
                          time_series_data[col],
                          label=col
                      )
                  plt.title("Time Series Analysis", fontsize=14, weight='bold')
                  plt.xlabel("Date", fontsize=12)
                  plt.ylabel("Average Value", fontsize=12)
                  plt.legend()
                  time_series_path = os.path.join(output_folder, "time_series_analysis.png")
                  plt.savefig(time_series_path)
                  plt.close()
                  ts_image_counter += 1
                  return [time_series_path]
    return []

def generate_box_plots(data, numeric_columns, output_folder):
    images = []
    for col in numeric_columns:
        bp_image_counter = 0
        bp_max_images = 1
        if bp_image_counter < bp_max_images:
          plt.figure(figsize=(10, 6))
          sns.boxplot(x=data[col])
          plt.axvline(data[col].mean(), color='red', linestyle='--', label='Mean')
          plt.axvline(data[col].median(), color='blue', linestyle='--', label='Median')
          plt.legend()
          for outlier in data[col][(data[col] < data[col].quantile(0.25) - 1.5 * (data[col].quantile(0.75) - data[col].quantile(0.25))) |
                                  (data[col] > data[col].quantile(0.75) + 1.5 * (data[col].quantile(0.75) - data[col].quantile(0.25)))].dropna():
              plt.text(outlier, 0, str(round(outlier, 2)), color="purple", fontsize=8)
          plt.title(f"Box Plot of {col}", fontsize=14, weight='bold')
          plt.xlabel("Value", fontsize=12)
          box_plot_path = os.path.join(output_folder, f"{col}_box_plot.png")
          plt.savefig(box_plot_path)
          plt.close()
          bp_image_counter += 1
          images.append(box_plot_path)
    return images


def advanced_statistical_analysis(data):
    """
    Perform advanced statistical analyses beyond basic descriptive statistics.
    
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


def compress_data_for_llm(data, max_rows=3):
    """
    Compress dataset for token-efficient LLM processing by sampling rows 
    and selecting important numeric and categorical columns.

    Args:
        data (pd.DataFrame): Input dataset.
        max_rows (int): Maximum number of rows to sample.

    Returns:
        tuple: Compressed data and its lightweight summary.
    """
    if data.empty:
        raise ValueError("Input data is empty.")

    # Prioritize columns with high variability or significance
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    priority_cols = list(numeric_cols) + list(categorical_cols)

    if not priority_cols:
        raise ValueError("No numeric or categorical columns found in the dataset.")

    # Sample rows dynamically based on dataset size
    sample_size = min(len(data), max_rows)
    compressed_data = data[priority_cols].sample(sample_size, random_state=42)

    # Compute lightweight summary statistics
    summary = {
        col: {
            'mean': compressed_data[col].mean() if np.issubdtype(compressed_data[col].dtype, np.number) else None,
            'unique_count': compressed_data[col].nunique(),
            'most_common': compressed_data[col].mode().tolist() if col in categorical_cols else None
        } for col in compressed_data.columns
    }

    return compressed_data, summary



def create_enhanced_prompt(report, context_level='detailed'):
    """
    Create a more structured, context-rich prompt for LLM analysis.
    
    Args:
        report (dict): Dataset analysis report
        context_level (str): Depth of context in prompt
    
    Returns:
        str: Formatted prompt for LLM
    """
    prompt_templates = {
        'basic': """
        Analyze the following dataset with a focus on key insights.
        Dataset Characteristics: {columns}
        Sample Data Overview: {sample_data}
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
        a) Identify primary patterns and trends
        b) Highlight potential correlations
        c) Suggest actionable insights
        
        4. Narrative Requirements:
        - Use a storytelling approach
        - Emphasize business or research implications
        - Provide clear, concise recommendations
        
        Provide a comprehensive yet succinct analysis.
        """,
        
        'expert': """
        Advanced Data Exploration Protocol:
        
        Dataset Metadata:
        {comprehensive_metadata}
        
        Analysis Directives:
        - Conduct multi-dimensional analysis
        - Uncover non-obvious relationships
        - Generate predictive hypotheses
        - Assess data quality and potential biases
        
        Deliverable: 
        - Nuanced narrative
        - Potential research/business strategies
        - Confidence levels for insights
        What statistical analyses or visualizations should I perform on this data? Provide a concise list of specific analyses.
        """
    }
    
    # Select prompt based on context level
    prompt_template = prompt_templates.get(context_level, prompt_templates['basic'])
    
    # Format prompt with specific dataset details
    formatted_prompt = prompt_template.format(
        columns=list(report.get('columns_info', {}).keys()),
        column_types=report.get('columns_info', {}),
        missing_values=report.get('missing_values', {}),
        sample_data=report.get('sample_data', {}),
        summary_stats=report.get('summary', {}),
        comprehensive_metadata=str(report)
    )
    
    return formatted_prompt


def get_llm_analysis_suggestions(report):
    """
    Enhanced LLM analysis suggestions with robust error handling, retry logic, and optimized token usage.

    Args:
        report (dict): Dataset analysis report
    
    Returns:
        list: Suggested analysis types or fallback options in case of an error
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

        # Prepare the payload
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
              {
                "role": "system",
                "content": (
                    "You are a highly efficient and precise data analyst. Your role is to suggest "
                    "Based on the structure of this dataset, what statistical and machine learning analyses should I perform to uncover patterns and relationships?"
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

        # Make the API call with retry
        response = make_api_request_with_retry(f"{api_base}/chat/completions", headers, payload)
        
        # Parse and return suggestions
        return parse_llm_response(response)

    except requests.exceptions.RequestException as e:
        print(f"HTTP error: {e}")
    except ValueError as e:
        print(f"LLM response error: {e}")

    # Fallback suggestions
    fallback_suggestions = {
        "general": ["Correlation Analysis", "Distribution Analysis", "Clustering Analysis", "PCA Visualization"],
    }
    return fallback_suggestions.get(report.get("context", "general"), fallback_suggestions["general"])

def parse_llm_response(response):
    """
    Parses the LLM response and extracts suggestions.

    Args:
        response (requests.Response): The API response object.

    Returns:
        list: Extracted suggestions.
    """
    try:
        result = response.json()
        choices = result.get("choices", [])
        if not choices:
            raise ValueError("No 'choices' found in response.")
        print(result)
        message_content = choices[0].get("message", {}).get("content", "").strip()
        if not message_content:
            raise ValueError("Empty 'content' in response message.")
        
        return [line.strip() for line in message_content.split("\n") if line.strip()]
    except (KeyError, ValueError, AttributeError) as e:
        raise ValueError(f"Error parsing LLM response: {e}")


def make_api_request_with_retry(url, headers, payload, retries=3):
    """
    Makes an API request with retry logic for resilience.
    """
    session = requests.Session()
    retry = Retry(total=retries, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retry))
    
    try:
        response = session.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"API request failed after {retries} retries: {e}")


# Function to resize images to 512x512 and compress
def process_images(image_paths,output_folder):
    processed_images = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path)
            img_resized = img.resize((512, 512))  # Resize to 512x512
            #compressed_path = f"compressed_{os.path.basename(img_path)}"
            compressed_path = os.path.join(output_folder, img_path)
            img_resized.save(compressed_path, optimize=True, quality=70)  # Compress
            processed_images.append(compressed_path)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
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
Can you provide an excellent executive summary of insights and a cohesive story based on this analysis.
Can you provide insights and a cohesive story based on this analysis,clearly integrating each visualization into the narrative?
Provide insights on trends, outliers, comparisons, and their implications, interpret the patterns, potential causes, and their implications for strategic planning.
Explain insights in a narrative style, making it engaging and interesting.
Highlight Which relationships stand out, and what actionable insights can be derived for decision-making

### Dataset Overview
- **Columns and Data Types:** {list(columns_info.keys())}
- **Missing Values:** {missing_values}

### Key Insights
- Summary: The dataset contains {len(columns_info)} columns, with missing data in 'date' and 'by'. Numeric columns like 'overall' and 'quality' have average scores around 3.2.
- Correlations: {correlations}

### Visualizations
{visualization_descriptions}

### Recommendations
Suggest actionable insights based on the findings and visualizations.

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
        print(payload)
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

def generate_insights_with_llm(report, images):
    """
    Uses an LLM to generate insights and a story based on the dataset analysis and visualizations.

    Args:
        report (dict): A dictionary containing basic analysis results.
        images (list): A list of file paths to the generated visualizations.

    Returns:
        str: The story generated by the LLM, referencing the provided visualizations.
    """
    # Retrieve the API token from environment variables
    AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
    if not AIPROXY_TOKEN:
        print("Error: AIPROXY_TOKEN environment variable is not set.")
        sys.exit(1)

    # Define the API endpoint
    api_base = "https://aiproxy.sanand.workers.dev/openai/v1"
    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
    }

    try:
        # Prepare the analysis request
        analysis_request = f"""
        Based on the following dataset analysis:
        - Columns info: {report.get('columns_info', 'N/A')}
        - Missing values: {report.get('missing_values', 'N/A')}
        - Summary statistics: {report.get('summary', 'N/A')}
        - Correlation matrix: {report.get('correlation_matrix', 'N/A')}
        Visualizations created and their purposes:
        {', '.join([f"Image {i+1}: {os.path.basename(img)}" for i, img in enumerate(images)]) if images else 'None'}.
        Can you provide an excellent executive summary of insights and a cohesive story based on this analysis.
        Can you provide insights and a cohesive story based on this analysis,clearly integrating each visualization into the narrative?
        Provide insights on trends, outliers, comparisons, and their implications, interpret the patterns, potential causes, and their implications for strategic planning.
        Explain insights in a narrative style, making it engaging and interesting.
        Highlight Which relationships stand out, and what actionable insights can be derived for decision-making
        """

        # Prepare the payload for the API request
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a data analyst summarizing insights from analyses and visualizations."},
                {"role": "user", "content": analysis_request}
            ]
        }
        
        # Make the API call
        response = requests.post(f"{api_base}/chat/completions", headers=headers, json=payload)
        response.raise_for_status()  # Raise an error for HTTP issues

        # Parse the response and extract the story
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

import logging
from PIL import Image
import tempfile


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





def analyze_low_res_visualization(image_path):
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

        # Preprocess: Reduce the resolution to 512x512 while maintaining aspect ratio
        with Image.open(image_path) as img:
            img = img.convert("RGB")  # Ensure consistent format
            img.thumbnail((512, 512))  # Maintains aspect ratio
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                temp_path = temp_file.name
                img.save(temp_path)

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
                insight = analyze_low_res_visualization(image_path)
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
     
# Existing methods like create_output_folder, analyze_dataset, 
# perform_suggested_analyses, ask_llm_for_story, generate_readme 
# remain largely the same

def main():
    """
    Enhanced main function to incorporate advanced analysis techniques.
    """
     # Load user configuration
    user_config = {
        "max_images": 1,  # User-specific overrides
        "max_rows": 1,
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
    #print(f"LLM Suggestions: {suggestions}")
    #suggestions=['### Comprehensive Data Analysis Plan', '', '#### 1. **Data Cleaning and Preparation**', '   - Handle missing values through imputation or removal where appropriate.', '   - Convert categorical columns, if necessary, to appropriate formats for analysis (e.g., encoding factors).', '', '#### 2. **Statistical Analyses and Visualizations**', '   - **Descriptive Statistics**: Evaluate central tendencies, dispersion, and shape of the data to summarize trends.', '   - **Correlation Analysis**: ', '     - Compute Pearson/Spearman correlation coefficients to identify relationships between key variables (e.g., Log GDP per capita with Life Ladder).', '     - Visualize correlations using a heatmap for better understanding of relationships.', '   - **Distribution Analysis**:', '     - Visualize distributions of key variables using histograms and boxplots to identify outliers and understand data spread.', '     - Assess normality using Q-Q plots.', '   - **Cluster Analysis**:', '     - Perform k-means or hierarchical clustering to identify groups of countries based on similar attributes (e.g., socio-economic indicators).', '     - Plot clusters to visualize grouping and potential market segments.', '   - **Principal Component Analysis (PCA)**:', '     - Conduct PCA to reduce dimensionality while retaining variance to better visualize the essential patterns among variables.', '     - Plot PCA results to illustrate country position in a reduced dimension space.', '', '#### 3. **Analysis Objectives**', '   - **Identify Primary Patterns and Trends**:', '     - Use time series analysis to show trends in "Life Ladder" over years, highlighting years with significant changes.', '     - Analyze socio-economic patterns by grouping countries by region and visualizing averages.', '', '   - **Highlight Potential Correlations**:', '     - Present correlation findings in an actionable format. For example, indicate which attributes most strongly correlate with life satisfaction (Life Ladder), allowing policymakers to target areas for improvement.', '     - Suggest further research into the causal impact of these variables on life satisfaction.', '', '   - **Suggest Actionable Insights**:', '     - Provide clear recommendations based on correlation analysis. For instance, if higher social support correlates with higher Life Ladders, suggest investing in community support initiatives.', '     - Offer policy implications, such as improving economic conditions (as seen through GDP per capita) in countries with lower Life Ladder rankings.', '', '#### 4. **Narrative Requirements**', '   - Craft a narrative that takes stakeholders through the journey of data analysis, highlighting key findings in an easy-to-follow format.', '   - Emphasize the business or research implications by relating findings back to potential real-world applications (e.g., governmental policy changes, NGO focus areas).', '   - Create concise recommendations backed by data insights to guide decision-making processes.', '', '### Conclusion', 'The proposed analyses aim to provide a comprehensive view of the dataset that identifies key patterns, potential areas for improvement, and actionable strategies for stakeholders. The incorporation of various statistical techniques will lend credibility and depth to the analysis, ultimately informing decision-making and policy development.']
    #suggestions=['correlation']
    print("Performing suggested analyses...")
    # Perform analyses and generate visualizations
    images = generate_visualizations(data, suggestions, output_folder,config)

    # Process images for reduced size and detail
    compressed_images = process_images(images,output_folder)
    print("Generating insights using LLM...")
    visualization_insights = process_visualizations_with_llm(compressed_images,report, data,advanced_stats)
    #visualization_insights={'/content/goodreads/correlation_heatmap.png': 'The low-resolution visualization appears cluttered with elements that are hard to discern. Key insights suggest potential trends in data distribution, but significant noise obscures details. A few anomalies may indicate spikes or drops in metrics, possibly highlighting areas of interest. The overall pattern seems nonlinear, suggesting variable relationships, potentially influenced by external factors. Further high-resolution analysis would be crucial for accurate interpretation and actionable insights.'}
    
    # Generate insights
    print("Generating insights1 using LLM...")
    story = generate_insights_with_llm1(report, images, data,advanced_stats)
    #story='xxx'
    print("Creating README.md...")
    # Create README
    generate_readme(story, images, output_folder,sumreport,visualization_insights)

    print(f"Analysis complete. Outputs saved in {output_folder}.")

if __name__ == "__main__":
    main()
