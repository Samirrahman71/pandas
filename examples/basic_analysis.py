#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic Sales Analysis Examples

This script demonstrates simple usage patterns for the SalesAnalyzer class.
"""

import sys
import os
import matplotlib.pyplot as plt

# Add parent directory to path to import sales_analytics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sales_analytics import SalesAnalyzer

def main():
    """
    Run basic analysis examples.
    """
    print("Basic Sales Analysis Examples")
    print("============================")
    
    # Check if sample data exists
    sample_file = '../sample_data/TableauSalesData.xlsx'
    if not os.path.exists(sample_file):
        print(f"Sample file not found: {sample_file}")
        print("Please provide path to your sales data file:")
        file_path = input("> ")
    else:
        file_path = sample_file
    
    # Initialize analyzer
    analyzer = SalesAnalyzer(file_path)
    print(f"Loaded data with {len(analyzer.data)} rows")
    
    # Example 1: Basic data cleaning
    print("\nExample 1: Data Cleaning")
    print("------------------------")
    clean_data = analyzer.clean_data()
    print(f"Data shape after cleaning: {clean_data.shape}")
    print(f"Data columns: {', '.join(clean_data.columns[:5])}...")
    
    # Example 2: Filtering data
    print("\nExample 2: Data Filtering")
    print("-------------------------")
    filtered_data = analyzer.get_filtered_data(region="West", category="Furniture")
    print(f"Filtered data shape: {filtered_data.shape}")
    print(f"Sample of filtered data:")
    if len(filtered_data) > 0:
        print(filtered_data[['Region', 'Category']].head())
    
    # Example 3: Profit analysis
    print("\nExample 3: Profit Analysis")
    print("--------------------------")
    try:
        profits, _ = analyzer.total_profits_by_subcategory()
        print("Top 3 most profitable sub-categories:")
        print(profits.head(3))
    except Exception as e:
        print(f"Could not run profit analysis: {str(e)}")
    
    # Example 4: Sales by region
    print("\nExample 4: Regional Analysis")
    print("---------------------------")
    try:
        regional_sales, _ = analyzer.sales_by_region()
        print("Sales by region:")
        print(regional_sales[['Region', 'Sales', 'Percentage']])
    except Exception as e:
        print(f"Could not run regional analysis: {str(e)}")
    
    print("\nAnalysis complete! For visualizations, uncomment the plt.show() line below.")
    # plt.show()  # Uncommented for interactive environments

if __name__ == "__main__":
    main()
