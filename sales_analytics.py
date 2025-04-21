#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tableau Sales Data Analysis Tool

This module provides comprehensive analytics and visualizations for sales data.
It leverages pandas for data manipulation and matplotlib/seaborn for visualization.

Author: Samir Rahman
GitHub: https://github.com/Samirrahman71
Date: April 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Union, Tuple
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('fivethirtyeight')
sns.set_palette("viridis")


class SalesAnalyzer:
    """
    A class for analyzing sales data with advanced pandas techniques.
    
    This class provides methods for data cleaning, transformation, and visualization
    to extract meaningful insights from sales datasets.
    """
    
    def __init__(self, file_path: str) -> None:
        """
        Initialize the SalesAnalyzer with a dataset.
        
        Parameters
        ----------
        file_path : str
            Path to the Excel or CSV file containing sales data
        """
        self.file_path = file_path
        self.data = self._load_data()
        self.cleaned_data = None
        # Store computed analytics for caching
        self._cache = {}
        
    def _load_data(self) -> pd.DataFrame:
        """
        Load the sales data from the provided file path.
        
        Returns
        -------
        pd.DataFrame
            The loaded sales data
        
        Raises
        ------
        FileNotFoundError
            If the file doesn't exist
        ValueError
            If the file format is not supported
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
            
        file_ext = os.path.splitext(self.file_path)[1].lower()
        
        if file_ext == '.xlsx' or file_ext == '.xls':
            return pd.read_excel(self.file_path)
        elif file_ext == '.csv':
            return pd.read_csv(self.file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}. Use .xlsx, .xls, or .csv")
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean the sales data by handling missing values, removing duplicates,
        and converting data types.
        
        Returns
        -------
        pd.DataFrame
            The cleaned sales data
        """
        if self.cleaned_data is not None:
            return self.cleaned_data
            
        df = self.data.copy()
        
        # Drop duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Fill missing numeric values with median
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        # Fill missing categorical values with mode
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
        
        # Convert date columns if they exist
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        for col in date_cols:
            if df[col].dtype == 'object':
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Create derived features
        if 'Order Date' in df.columns and pd.api.types.is_datetime64_dtype(df['Order Date']):
            df['Year'] = df['Order Date'].dt.year
            df['Month'] = df['Order Date'].dt.month
            df['Quarter'] = df['Order Date'].dt.quarter
            df['Day of Week'] = df['Order Date'].dt.day_name()
        
        self.cleaned_data = df
        return df
    
    def get_filtered_data(self, 
                         region: Optional[str] = None,
                         category: Optional[str] = None,
                         sub_category: Optional[str] = None,
                         product_name: Optional[str] = None) -> pd.DataFrame:
        """
        Filter the dataset based on provided criteria.
        
        Parameters
        ----------
        region : str, optional
            Region to filter by
        category : str, optional
            Category to filter by
        sub_category : str, optional
            Sub-category to filter by
        product_name : str, optional
            Product name to filter by
            
        Returns
        -------
        pd.DataFrame
            Filtered dataset
        """
        df = self.clean_data()
        
        if region and region != 'All' and 'Region' in df.columns:
            df = df[df['Region'] == region]
            
        if category and category != 'All' and 'Category' in df.columns:
            df = df[df['Category'] == category]
            
        if sub_category and sub_category != 'All' and 'Sub-Category' in df.columns:
            df = df[df['Sub-Category'] == sub_category]
            
        if product_name and product_name != 'All' and 'Product Name' in df.columns:
            df = df[df['Product Name'] == product_name]
            
        return df
    
    def total_profits_by_subcategory(self, 
                                    region: Optional[str] = None,
                                    category: Optional[str] = None) -> Tuple[pd.DataFrame, plt.Figure]:
        """
        Calculate total profits by sub-category.
        
        Parameters
        ----------
        region : str, optional
            Region to filter by
        category : str, optional
            Category to filter by
            
        Returns
        -------
        tuple
            (DataFrame with results, Matplotlib figure)
        """
        # Generate cache key
        cache_key = f"profits_subcategory_{region}_{category}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        df = self.get_filtered_data(region=region, category=category)
        
        if 'Profit' not in df.columns or 'Sub-Category' not in df.columns:
            raise ValueError("Data must contain 'Profit' and 'Sub-Category' columns")
        
        # Calculate total profits by sub-category
        profits_by_subcategory = df.groupby('Sub-Category')['Profit'].sum().sort_values(ascending=False).reset_index()
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x='Profit', y='Sub-Category', data=profits_by_subcategory, ax=ax)
        
        ax.set_title('Total Profits by Sub-Category', fontsize=16)
        ax.set_xlabel('Profit ($)', fontsize=14)
        ax.set_ylabel('Sub-Category', fontsize=14)
        
        # Add profit values to the end of each bar
        for i, profit in enumerate(profits_by_subcategory['Profit']):
            ax.text(profit + 10, i, f'${profit:,.2f}', va='center')
        
        plt.tight_layout()
        
        # Store in cache
        self._cache[cache_key] = (profits_by_subcategory, fig)
        
        return profits_by_subcategory, fig
    
    def least_profitable_products(self, 
                                 n: int = 10,
                                 region: Optional[str] = None,
                                 category: Optional[str] = None) -> Tuple[pd.DataFrame, plt.Figure]:
        """
        Identify the least profitable products.
        
        Parameters
        ----------
        n : int
            Number of least profitable products to show
        region : str, optional
            Region to filter by
        category : str, optional
            Category to filter by
            
        Returns
        -------
        tuple
            (DataFrame with results, Matplotlib figure)
        """
        # Generate cache key
        cache_key = f"least_profitable_{n}_{region}_{category}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        df = self.get_filtered_data(region=region, category=category)
        
        if 'Profit' not in df.columns or 'Product Name' not in df.columns:
            raise ValueError("Data must contain 'Profit' and 'Product Name' columns")
        
        # Calculate total profits by product
        profits_by_product = df.groupby('Product Name')['Profit'].sum().reset_index()
        
        # Get the n least profitable products
        least_profitable = profits_by_product.sort_values('Profit').head(n)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Use a red color palette for negative values
        palette = ["#FF9999" if x < 0 else "#FFD700" for x in least_profitable['Profit']]
        bars = sns.barplot(x='Profit', y='Product Name', data=least_profitable, palette=palette, ax=ax)
        
        ax.set_title(f'{n} Least Profitable Products', fontsize=16)
        ax.set_xlabel('Profit ($)', fontsize=14)
        ax.set_ylabel('Product Name', fontsize=14)
        
        # Add profit values to the end of each bar
        for i, profit in enumerate(least_profitable['Profit']):
            ax.text(profit - 100 if profit < 0 else profit + 10, 
                   i, 
                   f'${profit:,.2f}', 
                   va='center', 
                   ha='right' if profit < 0 else 'left',
                   color='black')
        
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.7)
        plt.tight_layout()
        
        # Store in cache
        self._cache[cache_key] = (least_profitable, fig)
        
        return least_profitable, fig
    
    def profits_for_least_profitable_subcategories(self, 
                                                 n: int = 3,
                                                 region: Optional[str] = None) -> Tuple[pd.DataFrame, plt.Figure]:
        """
        Calculate total profits for the least profitable sub-categories.
        
        Parameters
        ----------
        n : int
            Number of least profitable sub-categories to analyze
        region : str, optional
            Region to filter by
            
        Returns
        -------
        tuple
            (DataFrame with results, Matplotlib figure)
        """
        # Generate cache key
        cache_key = f"least_profitable_subcategories_{n}_{region}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        df = self.get_filtered_data(region=region)
        
        if 'Profit' not in df.columns or 'Sub-Category' not in df.columns:
            raise ValueError("Data must contain 'Profit' and 'Sub-Category' columns")
        
        # Calculate total profits by sub-category
        profits_by_subcategory = df.groupby('Sub-Category')['Profit'].sum().reset_index()
        
        # Get the n least profitable sub-categories
        least_profitable = profits_by_subcategory.sort_values('Profit').head(n)
        
        # Filter original data to only include these sub-categories
        filtered_df = df[df['Sub-Category'].isin(least_profitable['Sub-Category'])]
        
        # Group by Sub-Category and Product Name to get detailed breakdown
        detailed_profits = filtered_df.groupby(['Sub-Category', 'Product Name'])['Profit'].sum().reset_index()
        detailed_profits = detailed_profits.sort_values(['Sub-Category', 'Profit'])
        
        # Create visualization - multiple plots
        fig, axes = plt.subplots(n, 1, figsize=(12, 5*n))
        
        if n == 1:
            axes = [axes]  # Make it iterable for single plot case
            
        for i, subcategory in enumerate(least_profitable['Sub-Category']):
            # Filter data for this subcategory
            subcat_data = detailed_profits[detailed_profits['Sub-Category'] == subcategory]
            subcat_data = subcat_data.sort_values('Profit').head(10)  # Top 10 least profitable products
            
            # Create bar plot
            palette = ["#FF9999" if x < 0 else "#FFD700" for x in subcat_data['Profit']]
            sns.barplot(x='Profit', y='Product Name', data=subcat_data, palette=palette, ax=axes[i])
            
            total_profit = least_profitable.loc[least_profitable['Sub-Category'] == subcategory, 'Profit'].values[0]
            axes[i].set_title(f'Sub-Category: {subcategory} (Total Profit: ${total_profit:,.2f})', fontsize=14)
            axes[i].set_xlabel('Profit ($)', fontsize=12)
            axes[i].set_ylabel('Product Name', fontsize=12)
            
            # Add profit values
            for j, profit in enumerate(subcat_data['Profit']):
                axes[i].text(profit - 10 if profit < 0 else profit + 10, 
                           j, 
                           f'${profit:,.2f}', 
                           va='center',
                           ha='right' if profit < 0 else 'left')
            
            axes[i].axvline(x=0, color='black', linestyle='-', alpha=0.7)
        
        plt.tight_layout()
        
        # Store in cache
        self._cache[cache_key] = (detailed_profits, fig)
        
        return detailed_profits, fig
    
    def sales_by_region(self) -> Tuple[pd.DataFrame, plt.Figure]:
        """
        Analyze sales distribution by region.
        
        Returns
        -------
        tuple
            (DataFrame with results, Matplotlib figure)
        """
        # Generate cache key
        cache_key = "sales_by_region"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        df = self.clean_data()
        
        if 'Sales' not in df.columns or 'Region' not in df.columns:
            raise ValueError("Data must contain 'Sales' and 'Region' columns")
        
        # Calculate sales by region
        sales_by_region = df.groupby('Region')['Sales'].sum().sort_values(ascending=False).reset_index()
        
        # Add percentage calculation
        total_sales = sales_by_region['Sales'].sum()
        sales_by_region['Percentage'] = (sales_by_region['Sales'] / total_sales * 100).round(2)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Bar chart
        sns.barplot(x='Region', y='Sales', data=sales_by_region, ax=ax1)
        ax1.set_title('Total Sales by Region', fontsize=16)
        ax1.set_xlabel('Region', fontsize=14)
        ax1.set_ylabel('Sales ($)', fontsize=14)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add sales values on top of bars
        for i, sales in enumerate(sales_by_region['Sales']):
            ax1.text(i, sales + 1000, f'${sales:,.0f}', ha='center')
        
        # Pie chart
        explode = [0.1 if i == 0 else 0 for i in range(len(sales_by_region))]  # Explode the largest slice
        ax2.pie(sales_by_region['Sales'], 
               labels=sales_by_region['Region'], 
               autopct='%1.1f%%',
               startangle=90,
               explode=explode,
               shadow=True)
        ax2.set_title('Sales Distribution by Region', fontsize=16)
        
        plt.tight_layout()
        
        # Store in cache
        self._cache[cache_key] = (sales_by_region, fig)
        
        return sales_by_region, fig
    
    def profit_trends_over_time(self) -> Tuple[pd.DataFrame, plt.Figure]:
        """
        Analyze profit trends over time.
        
        Returns
        -------
        tuple
            (DataFrame with results, Matplotlib figure)
        """
        # Generate cache key
        cache_key = "profit_trends"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        df = self.clean_data()
        
        if 'Profit' not in df.columns or 'Order Date' not in df.columns:
            raise ValueError("Data must contain 'Profit' and 'Order Date' columns")
        
        # Ensure Order Date is datetime
        if not pd.api.types.is_datetime64_dtype(df['Order Date']):
            df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
        
        # Create monthly time series
        df['YearMonth'] = df['Order Date'].dt.to_period('M')
        monthly_profits = df.groupby('YearMonth')['Profit'].sum().reset_index()
        monthly_profits['YearMonth'] = monthly_profits['YearMonth'].astype(str)
        
        # Calculate moving average (trailing 3 months)
        monthly_profits['3-Month MA'] = monthly_profits['Profit'].rolling(window=3).mean()
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot lines
        ax.plot(monthly_profits['YearMonth'], monthly_profits['Profit'], marker='o', label='Monthly Profit')
        ax.plot(monthly_profits['YearMonth'], monthly_profits['3-Month MA'], linestyle='--', linewidth=2, label='3-Month Moving Average')
        
        # Highlight trend
        z = np.polyfit(range(len(monthly_profits)), monthly_profits['Profit'], 1)
        p = np.poly1d(z)
        ax.plot(monthly_profits['YearMonth'], p(range(len(monthly_profits))), "r--", linewidth=1, label='Trend Line')
        
        ax.set_title('Profit Trends Over Time', fontsize=16)
        ax.set_xlabel('Year-Month', fontsize=14)
        ax.set_ylabel('Profit ($)', fontsize=14)
        ax.tick_params(axis='x', rotation=90)
        ax.legend()
        
        # Add horizontal line at zero profit
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Highlight periods of profit decline in red background
        for i in range(1, len(monthly_profits)):
            if monthly_profits['Profit'].iloc[i] < monthly_profits['Profit'].iloc[i-1]:
                ax.axvspan(i-1, i, alpha=0.2, color='red')
        
        plt.tight_layout()
        
        # Store in cache
        self._cache[cache_key] = (monthly_profits, fig)
        
        return monthly_profits, fig
    
    def customer_segment_analysis(self) -> Tuple[pd.DataFrame, plt.Figure]:
        """
        Analyze sales and profits by customer segment.
        
        Returns
        -------
        tuple
            (DataFrame with results, Matplotlib figure)
        """
        # Generate cache key
        cache_key = "customer_segments"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        df = self.clean_data()
        
        if 'Segment' not in df.columns or 'Sales' not in df.columns or 'Profit' not in df.columns:
            raise ValueError("Data must contain 'Segment', 'Sales', and 'Profit' columns")
        
        # Calculate metrics by segment
        segment_analysis = df.groupby('Segment').agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Order ID': 'nunique',  # Number of orders
            'Customer Name': 'nunique',  # Number of customers
        }).reset_index()
        
        # Calculate profit margin
        segment_analysis['Profit Margin'] = (segment_analysis['Profit'] / segment_analysis['Sales'] * 100).round(2)
        # Calculate average order value
        segment_analysis['Avg Order Value'] = (segment_analysis['Sales'] / segment_analysis['Order ID']).round(2)
        
        # Sort by profit
        segment_analysis = segment_analysis.sort_values('Profit', ascending=False)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        # Sales by segment
        sns.barplot(x='Segment', y='Sales', data=segment_analysis, ax=axes[0, 0])
        axes[0, 0].set_title('Sales by Customer Segment', fontsize=14)
        axes[0, 0].set_xlabel('Segment', fontsize=12)
        axes[0, 0].set_ylabel('Sales ($)', fontsize=12)
        
        # Profit by segment
        sns.barplot(x='Segment', y='Profit', data=segment_analysis, ax=axes[0, 1])
        axes[0, 1].set_title('Profit by Customer Segment', fontsize=14)
        axes[0, 1].set_xlabel('Segment', fontsize=12)
        axes[0, 1].set_ylabel('Profit ($)', fontsize=12)
        
        # Profit margin by segment
        sns.barplot(x='Segment', y='Profit Margin', data=segment_analysis, ax=axes[1, 0])
        axes[1, 0].set_title('Profit Margin by Customer Segment', fontsize=14)
        axes[1, 0].set_xlabel('Segment', fontsize=12)
        axes[1, 0].set_ylabel('Profit Margin (%)', fontsize=12)
        
        # Average order value by segment
        sns.barplot(x='Segment', y='Avg Order Value', data=segment_analysis, ax=axes[1, 1])
        axes[1, 1].set_title('Average Order Value by Customer Segment', fontsize=14)
        axes[1, 1].set_xlabel('Segment', fontsize=12)
        axes[1, 1].set_ylabel('Average Order Value ($)', fontsize=12)
        
        plt.tight_layout()
        
        # Store in cache
        self._cache[cache_key] = (segment_analysis, fig)
        
        return segment_analysis, fig


def demo_sales_analyzer():
    """
    Run a demonstration of the SalesAnalyzer class.
    
    This function shows how to use the SalesAnalyzer class with example data.
    """
    # Example usage - if running with sample data
    print("SalesAnalyzer Demo")
    print("=================")
    print("Note: This demo requires a TableauSalesData.xlsx file")
    print("If you don't have this file, please use your own sales data file")
    print()
    
    try:
        # Try to find the sample data file
        file_path = 'TableauSalesData.xlsx'
        if not os.path.exists(file_path):
            print(f"Sample file {file_path} not found.")
            print("Please provide the path to your sales data file:")
            file_path = input("> ")
            
        # Initialize the analyzer
        analyzer = SalesAnalyzer(file_path)
        print(f"Successfully loaded data with {len(analyzer.data)} rows and {len(analyzer.data.columns)} columns")
        
        # Display sample analyses
        analyses = [
            ("Total Profits by Sub-Category", analyzer.total_profits_by_subcategory),
            ("10 Least Profitable Products", lambda: analyzer.least_profitable_products(10)),
            ("Profits for 3 Least Profitable Sub-Categories", 
             lambda: analyzer.profits_for_least_profitable_subcategories(3)),
            ("Sales by Region", analyzer.sales_by_region),
            ("Profit Trends Over Time", analyzer.profit_trends_over_time),
            ("Customer Segment Analysis", analyzer.customer_segment_analysis)
        ]
        
        for name, analysis_func in analyses:
            print(f"\nPerforming analysis: {name}")
            try:
                results, fig = analysis_func()
                print(f"Analysis complete - generated visualization")
                
                # Display the first few rows of results
                if isinstance(results, pd.DataFrame) and not results.empty:
                    print("\nSample results:")
                    print(results.head().to_string())
                    
                # For a real interactive session, you might want to display the figure
                # plt.show() - commented out for non-interactive environments
                
            except Exception as e:
                print(f"Error during analysis: {str(e)}")
                
    except Exception as e:
        print(f"Error in demo: {str(e)}")


if __name__ == "__main__":
    demo_sales_analyzer()
