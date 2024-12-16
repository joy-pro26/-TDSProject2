
# Autolysis: Automated Data Analysis Tool

## Introduction

Autolysis is a powerful tool designed to simplify data analysis and visualization, transforming raw datasets into actionable insights. By combining advanced statistical analysis, machine learning, and the interpretive capabilities of large language models (LLMs), Autolysis enables users to uncover patterns, trends, and outliers in their data with minimal effort. This tool is ideal for data scientists, business analysts, and researchers looking to streamline their workflow and generate impactful reports.

---

## Key Features

- **Automated Data Exploration**: Generates detailed summaries, correlation matrices, and statistical insights.
- **Advanced Statistical Analysis**: Includes normality tests, outlier detection, and correlation analyses (Spearman and Kendall).
- **Dynamic Visualizations**: Creates visualizations such as heatmaps, PCA plots, distribution plots, and time series graphs.
- **Machine Learning Integration**: Implements clustering (KMeans) and dimensionality reduction (PCA).
- **LLM-Powered Insights**: Produces narrative-style insights and recommendations based on the analysis.
- **Comprehensive Reporting**: Consolidates all findings, including visualizations, into a professional-grade `README.md` file.

---

## Workflow

1. **Input**: Provide a CSV dataset.
2. **Analysis**: The tool performs:
   - Data validation and cleaning.
   - Summary statistics generation.
   - Advanced statistical analyses.
3. **Visualization**: Generates graphs, charts, and visual insights.
4. **LLM Insights**: Leverages GPT-powered analysis for storytelling and recommendations.
5. **Output**: All results are saved in a structured `README.md` file along with visualizations.

---

## Usage

1. Place the dataset (CSV file) in the same directory as the script.
2. Run the script:
   ```bash
   python autolysis.py <dataset.csv>
   ```
3. Outputs, including visualizations and the `README.md`, will be saved in a folder named after the dataset.

---

## Visualizations

Autolysis generates a variety of visualizations, such as:

- **Heatmaps**: To display correlations between numeric columns.
- **Box Plots**: To detect outliers and visualize distributions.
- **PCA Plots**: To reduce dimensionality and visualize clusters.
- **Time Series Graphs**: To analyze trends over time.
- **Cluster Scatterplots**: To show KMeans clustering results.

### Examples

Below are examples of visualizations generated:

&#x20;

---

## Advanced Statistical Insights

1. **Normality Tests**: Determines if numeric columns follow a normal distribution.
2. **Outlier Detection**:
   - **IQR Method**: Counts of outliers based on interquartile range.
   - **Z-Score Method**: Identifies extreme values.
3. **Correlations**:
   - **Spearman Correlation**: Measures monotonic relationships between variables.
   - **Kendall Correlation**: Measures ordinal associations.

---

## LLM-Powered Insights

The tool integrates with OpenAI's GPT-4o-mini model to generate:

- **Key insights**: Trends, patterns, and anomalies.
- **Actionable recommendations**: Strategic steps based on data analysis.
- **Visual interpretations**: Narrative descriptions of generated charts and graphs.

---

## Sample Output

### **Executive Summary**

> "The dataset reveals strong correlations between `overall` and `quality` (Spearman: 0.82). Distribution analysis highlights significant outliers in the `score` variable, suggesting possible data errors or anomalies. Clustering identified three distinct customer segments, with PCA revealing two key dimensions driving variation. Recommendations include focusing on high-quality scores for targeted marketing and addressing inconsistencies in data collection."

---

## Appendix

### **Summary Statistics**

```plaintext
| Column         | Mean   | Std Dev | Min   | Max   |
|----------------|--------|---------|-------|-------|
| Quality Score  | 4.2    | 0.8     | 1.0   | 5.0   |
| Overall Rating | 4.0    | 1.1     | 1.0   | 5.0   |
```

### **Correlation Matrix**

```plaintext
|              | Quality | Overall |
|--------------|---------|---------|
| Quality      | 1.0     | 0.82    |
| Overall      | 0.82    | 1.0     |
```

### **Covariance Matrix**

```plaintext
|              | Quality | Overall |
|--------------|---------|---------|
| Quality      | 0.64    | 0.55    |
| Overall      | 0.55    | 1.21    |
```

---

## Configuration

The tool supports customizable configurations:

- `max_images`: Limit the number of visualizations generated.
- `date_column`: Specify the column for time series analysis.
- `n_clusters`: Number of clusters for KMeans analysis.
- `max_rows`: Row limit for memory efficiency.

---

## Requirements

- Python 3.8 or above.
- Required libraries:
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn scipy openai pillow
  ```

---

## Contact

For questions or feedback, reach out to the development team at [[your-email@example.com](mailto\:your-email@example.com)].
