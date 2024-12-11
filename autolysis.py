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
from scipy.stats import spearmanr, kendalltau
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

def generate_visualizations(data, suggestions, output_folder):
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

    max_images = 6  # Limit the number of generated PNG files
    image_counter = 0
    cor_max_images = 1
    dist_max_images = 1
    dist_image_counter = 0
    clust_max_images = 1
    clust_image_counter = 0
    pca_max_images = 1
    pca_image_counter = 0
    ts_image_counter = 0
    ts_max_images = 1
    bp_image_counter = 0
    bp_max_images = 1

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
        plt.figure(figsize=(10, 8))
        sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm", cbar_kws={'label': 'Correlation Coefficient'})
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
    cor_image_counter = 0
    cor_max_images = 1
    for col in numeric_columns:
        if data[col].dropna().shape[0] > 0:
            if cor_image_counter < cor_max_images:
              col_range = data[col].max() - data[col].min()
              if col_range > 0:
                  plt.figure(figsize=(10, 6))
                  sns.histplot(data[col], kde=True, binwidth=col_range / 30)
                  plt.title(f"Distribution of {col}", fontsize=14, weight='bold')
                  plt.xlabel("Value", fontsize=12)
                  plt.ylabel("Frequency", fontsize=12)
                  hist_path = os.path.join(output_folder, f"{col}_distribution.png")
                  plt.savefig(hist_path)
                  plt.close()
                  cor_image_counter+= 1
                  images.append(hist_path)
    return images

def generate_cluster_visualization(data, numeric_columns, output_folder):
    if numeric_columns.shape[1] > 1:
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
        return [cluster_path]
    return []

def generate_pca_visualization(data, numeric_columns, output_folder):
    if numeric_columns.shape[1] > 1:
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
        return [pca_path]
    return []

def generate_time_series_plot(data, numeric_columns, output_folder):
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
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
                return [time_series_path]
    return []

def generate_box_plots(data, numeric_columns, output_folder):
    images = []
    for col in numeric_columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=data[col])
        plt.title(f"Box Plot of {col}", fontsize=14, weight='bold')
        plt.xlabel("Value", fontsize=12)
        box_plot_path = os.path.join(output_folder, f"{col}_box_plot.png")
        plt.savefig(box_plot_path)
        plt.close()
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
    Compress dataset for more token-efficient LLM processing by sampling rows 
    and selecting all numeric and categorical columns.
    
    Args:
        data (pd.DataFrame): Input dataset
        max_rows (int): Maximum number of rows to sample
    
    Returns:
        tuple: Compressed data and its summary
    """
    if data.empty:
        raise ValueError("Input data is empty.")
    
       
    # Select numeric and categorical columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    
    # Combine all selected columns
    priority_cols = list(numeric_cols) + list(categorical_cols)
    
    if len(priority_cols) == 0:
        raise ValueError("No numeric or categorical columns found in the dataset.")
    
    # Sample rows
    compressed_data = data[priority_cols].sample(min(len(data), max_rows), random_state=42)
    
    # Compute lightweight summary statistics
    summary = {
        col: {
            'mean': compressed_data[col].mean() if np.issubdtype(compressed_data[col].dtype, np.number) else None,
            'unique_count': compressed_data[col].nunique(),
            'most_common': compressed_data[col].mode().tolist()
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

def generate_insights_with_llm(report, images, data):
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
    analysis_request = f"""
You are a data analyst tasked with generating a comprehensive and coherent narrative report based on the following data analysis results.
### Executive summary
Provide an excellent executive summary of insights and a cohesive story based on this analysis.
### Introduction
Provide a brief overview of the dataset, including its context and the purpose of the analysis.

### Dataset Overview
- **Columns and Data Types:** {columns_info}
- **Missing Values:** {missing_values}

### Key Findings
#### Summary Statistics
Analyze the following summary statistics and highlight key insights:
{summary_stats}

#### Correlation Analysis
Discuss significant correlations found in the data:
{advanced_stats.get('advanced_correlations', {})}

### Visualizations
Integrate the following visualizations into your narrative, explaining what each illustrates:
{visualization_descriptions}

### Insights and Implications
Interpret the findings, explaining the significance of trends, patterns, and outliers. Discuss potential causes and implications for stakeholders.

### Recommendations
Based on the analysis, provide actionable recommendations or strategies.

### Conclusion
Summarize the key takeaways from the analysis.

**Note:** Ensure the narrative flows logically, is engaging, and uses a storytelling approach. Reference the visualizations appropriately within the narrative.
"""

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

def analyze_visualization_with_llm(image_path):
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()

    try:
        # Prepare the payload
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are an expert data analyst and storyteller."},
                {"role": "user", "content": "Analyze this visualization for insights."}
            ]
        }

        # Make the API call
        api_base = "https://aiproxy.sanand.workers.dev/openai/v1"
        headers = {
            "Authorization": f"Bearer {os.environ.get('AIPROXY_TOKEN')}",
            "Content-Type": "application/json"
        }
        response = requests.post(
            f"{api_base}/chat/completions",
            headers=headers,
            json=payload,
            files={"file": (os.path.basename(image_path), image_bytes)}
        )
        response.raise_for_status()

        # Extract the insights
        result = response.json()
        insight = result["choices"][0]["message"]["content"].strip()
        return insight

    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
        return ""
    except KeyError as e:
        print(f"Unexpected response structure: {e}")
        return ""

def process_visualizations_with_llm(images):
    insights = []
    for img in images:
        insight = analyze_visualization_with_llm(img)
        if insight:
            insights.append((img, insight))
    return insights

        
def generate_readme(story, images,insights,output_folder,report):
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
        for img, insight in insights:
            f.write(f"### Insight from {os.path.basename(img)}\n")
            f.write(f"{insight}\n\n")
            f.write(f"![{os.path.basename(img)}]({os.path.basename(img)})\n\n")
                # Summary Statistics
        f.write("## Appendix \n")
        f.write("## Summary Statistics of the given data\n")
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
    #suggestions=['clustering']
    print("Performing suggested analyses...")
    # Perform analyses and generate visualizations
    images = generate_visualizations(data, suggestions, output_folder)

    # Process images for reduced size and detail
    #compressed_images = process_images(images,output_folder)
    print("Generating insights using LLM...")
    insights = process_visualizations_with_llm(images)
    # Generate insights
    story = generate_insights_with_llm(report, images,data)
    #story='xxx'
    print("Creating README.md...")
    # Create README
    generate_readme(story, images,insights, output_folder,sumreport)

    print(f"Analysis complete. Outputs saved in {output_folder}.")

if __name__ == "__main__":
    main()
