#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Sales Analytics Techniques

This script demonstrates more complex usage patterns and advanced pandas
techniques using the SalesAnalyzer class.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict

# Add parent directory to path to import sales_analytics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sales_analytics import SalesAnalyzer

def demonstrate_custom_aggregations(analyzer: SalesAnalyzer) -> pd.DataFrame:
    """
    Demonstrate custom aggregation techniques with pandas.
    
    Parameters
    ----------
    analyzer : SalesAnalyzer
        Initialized SalesAnalyzer instance
        
    Returns
    -------
    pd.DataFrame
        Aggregated data result
    """
    print("Custom Aggregation Example")
    print("-------------------------")
    
    # Get clean data
    df = analyzer.clean_data()
    
    # Define custom aggregation functions
    def profit_margin(x):
        """Calculate profit margin"""
        return (x['Profit'].sum() / x['Sales'].sum() * 100).round(2)
    
    def count_high_value(x):
        """Count orders above $1000"""
        return (x > 1000).sum()
    
    # Multi-level aggregation with custom functions
    agg_result = df.groupby(['Region', 'Category']).agg({
        'Sales': ['sum', 'mean', 'median', lambda x: count_high_value(x)],
        'Profit': ['sum', 'mean'],
        'Customer ID': 'nunique',
        'Order ID': 'count'
    })
    
    # Rename columns for clarity
    agg_result.columns = ['_'.join(col).strip() for col in agg_result.columns.values]
    agg_result.rename(columns={
        'Sales_<lambda>': 'High_Value_Orders',
        'Customer ID_nunique': 'Unique_Customers',
        'Order ID_count': 'Total_Orders'
    }, inplace=True)
    
    # Add calculated columns
    agg_result['Profit_Margin'] = (agg_result['Profit_sum'] / agg_result['Sales_sum'] * 100).round(2)
    agg_result['Avg_Order_Value'] = (agg_result['Sales_sum'] / agg_result['Total_Orders']).round(2)
    agg_result['Orders_Per_Customer'] = (agg_result['Total_Orders'] / agg_result['Unique_Customers']).round(2)
    
    # Sort by total sales descending
    agg_result = agg_result.sort_values('Sales_sum', ascending=False)
    
    print(f"\nTop 5 region-category combinations by sales:")
    print(agg_result[['Sales_sum', 'Profit_Margin', 'Unique_Customers']].head(5))
    
    return agg_result
    
def demonstrate_time_series_techniques(analyzer: SalesAnalyzer) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Demonstrate advanced time series analysis techniques.
    
    Parameters
    ----------
    analyzer : SalesAnalyzer
        Initialized SalesAnalyzer instance
        
    Returns
    -------
    tuple
        (DataFrame with results, Matplotlib figure)
    """
    print("\nAdvanced Time Series Analysis")
    print("---------------------------")
    
    # Get clean data
    df = analyzer.clean_data()
    
    # Ensure we have date information
    if 'Order Date' not in df.columns:
        print("Error: Data doesn't contain 'Order Date' column")
        return pd.DataFrame(), None
    
    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_dtype(df['Order Date']):
        df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    
    # Create monthly time series
    df['Year-Month'] = df['Order Date'].dt.to_period('M')
    monthly_data = df.groupby('Year-Month').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order ID': 'nunique',
        'Customer ID': 'nunique'
    }).reset_index()
    monthly_data['Year-Month'] = monthly_data['Year-Month'].astype(str)
    
    # Calculate rolling statistics
    monthly_data['Sales_3MA'] = monthly_data['Sales'].rolling(window=3).mean()
    monthly_data['Sales_YoY'] = monthly_data['Sales'].pct_change(12) * 100
    
    # Calculate cumulative metrics
    monthly_data['Cumulative_Sales'] = monthly_data['Sales'].cumsum()
    
    # Create correlation features
    monthly_data['Sales_vs_Orders'] = monthly_data['Sales'] / monthly_data['Order ID']
    
    # Implement seasonal decomposition (if we have enough data)
    if len(monthly_data) >= 12:
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            # Create a copy with datetime index for decomposition
            ts_data = monthly_data.copy()
            ts_data['Date'] = pd.to_datetime(ts_data['Year-Month'])
            ts_data.set_index('Date', inplace=True)
            
            # Perform decomposition
            decomposition = seasonal_decompose(ts_data['Sales'], model='multiplicative', period=12)
            
            # Create visualization
            fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
            decomposition.observed.plot(ax=axes[0], legend=False)
            axes[0].set_title('Observed')
            decomposition.trend.plot(ax=axes[1], legend=False)
            axes[1].set_title('Trend')
            decomposition.seasonal.plot(ax=axes[2], legend=False)
            axes[2].set_title('Seasonality')
            decomposition.resid.plot(ax=axes[3], legend=False)
            axes[3].set_title('Residuals')
            
            plt.tight_layout()
            
            print(f"Performed seasonal decomposition on {len(monthly_data)} months of data")
            print("Trend, seasonality, and residual components extracted")
            
            return monthly_data, fig
            
        except Exception as e:
            print(f"Couldn't perform seasonal decomposition: {str(e)}")
    
    # If seasonal decomposition failed, create a simple visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(monthly_data['Year-Month'], monthly_data['Sales'], marker='o', label='Monthly Sales')
    ax.plot(monthly_data['Year-Month'], monthly_data['Sales_3MA'], linestyle='--', label='3-Month Moving Avg')
    
    ax.set_title('Sales Time Series Analysis', fontsize=16)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Sales ($)', fontsize=12)
    ax.tick_params(axis='x', rotation=90)
    ax.legend()
    
    plt.tight_layout()
    
    print(f"Time series analysis complete for {len(monthly_data)} months")
    print(f"Last 5 months of data:")
    print(monthly_data[['Year-Month', 'Sales', 'Sales_3MA', 'Sales_YoY']].tail(5))
    
    return monthly_data, fig

def demonstrate_customer_cohort_analysis(analyzer: SalesAnalyzer) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Demonstrate customer cohort analysis techniques.
    
    Parameters
    ----------
    analyzer : SalesAnalyzer
        Initialized SalesAnalyzer instance
        
    Returns
    -------
    tuple
        (DataFrame with results, Matplotlib figure)
    """
    print("\nCustomer Cohort Analysis")
    print("----------------------")
    
    # Get clean data
    df = analyzer.clean_data()
    
    # Check required columns
    required_cols = ['Customer ID', 'Order Date', 'Sales']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Data missing required columns: {required_cols}")
        return pd.DataFrame(), None
    
    # Ensure data types
    if not pd.api.types.is_datetime64_dtype(df['Order Date']):
        df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    
    # Extract cohort information
    df['Cohort Month'] = df['Order Date'].dt.to_period('M')
    
    # Get the first purchase month for each customer
    first_purchase = df.groupby('Customer ID')['Cohort Month'].min().reset_index()
    first_purchase.rename(columns={'Cohort Month': 'First Purchase'}, inplace=True)
    
    # Merge with original data
    df = df.merge(first_purchase, on='Customer ID')
    
    # Calculate time difference (in months) from first purchase
    df['Months Since First Purchase'] = ((df['Cohort Month'].dt.year - df['First Purchase'].dt.year) * 12 + 
                                        (df['Cohort Month'].dt.month - df['First Purchase'].dt.month))
    
    # Create cohort analysis
    cohort_data = df.groupby(['First Purchase', 'Months Since First Purchase']).agg({
        'Customer ID': 'nunique',
        'Sales': 'sum'
    }).reset_index()
    
    # Calculate retention rates
    cohort_sizes = cohort_data[cohort_data['Months Since First Purchase'] == 0].set_index('First Purchase')['Customer ID']
    cohort_data['Cohort Size'] = cohort_data['First Purchase'].map(cohort_sizes)
    cohort_data['Retention Rate'] = (cohort_data['Customer ID'] / cohort_data['Cohort Size'] * 100).round(2)
    
    # Create pivot table for retention visualization
    retention_pivot = cohort_data.pivot_table(index='First Purchase', 
                                             columns='Months Since First Purchase', 
                                             values='Retention Rate')
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(retention_pivot, annot=True, fmt='.1f', cmap='YlGnBu', ax=ax)
    
    ax.set_title('Customer Retention by Cohort (%)', fontsize=16)
    ax.set_ylabel('Cohort (First Purchase Month)', fontsize=12)
    ax.set_xlabel('Months Since First Purchase', fontsize=12)
    
    plt.tight_layout()
    
    print(f"Cohort analysis complete for {len(cohort_sizes)} customer cohorts")
    print(f"Average retention rate after 3 months: {cohort_data[cohort_data['Months Since First Purchase'] == 3]['Retention Rate'].mean():.2f}%")
    
    return cohort_data, fig

def main():
    """
    Run advanced analysis examples.
    """
    print("Advanced Sales Analytics Techniques")
    print("==================================")
    
    # Check if sample data exists
    sample_file = '../sample_data/TableauSalesData.xlsx'
    if not os.path.exists(sample_file):
        print(f"Sample file not found: {sample_file}")
        print("Please provide path to your sales data file:")
        file_path = input("> ")
    else:
        file_path = sample_file
    
    # Initialize analyzer
    try:
        analyzer = SalesAnalyzer(file_path)
        print(f"Loaded data with {len(analyzer.data)} rows and {len(analyzer.data.columns)} columns")
        
        # Run advanced techniques
        agg_data = demonstrate_custom_aggregations(analyzer)
        time_series_data, ts_fig = demonstrate_time_series_techniques(analyzer)
        cohort_data, cohort_fig = demonstrate_customer_cohort_analysis(analyzer)
        
        print("\nAnalysis complete! For visualizations, uncomment the plt.show() line below.")
        # plt.show()  # Uncomment for interactive environments
        
    except Exception as e:
        print(f"Error in advanced analysis: {str(e)}")

if __name__ == "__main__":
    main()
