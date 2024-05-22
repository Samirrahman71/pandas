# pandas
Utilized Pandas to extract and highlight six key analytics from a Tableau sales dataset. This involved data cleaning, transformation, and the creation of summary statistics to provide insights into sales performance and trends.
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import ipywidgets as widgets

# Load the spreadsheet data
data = pd.read_excel('/content/TableauSalesData.xlsx')

# Define widgets
region_dd = widgets.Dropdown(options=['All'] + sorted(data['Region'].unique()), description="Region:")
category_dd = widgets.Dropdown(description="Category:")
sub_category_dd = widgets.Dropdown(description="Sub-Category:")
product_name_dd = widgets.Dropdown(description="Product Name:")
analytics_dd = widgets.Dropdown(
    options=[
        'Total Profits by Sub-Category',
        '10 Least Profitable Products',
        'Total Profits for 3 Least Profitable Sub-Categories'
    ],
    description="Analytics:"
)
exit_button = widgets.Button(description="Exit")
output = widgets.Output()

# Update functions for dropdowns
def update_category(*args):
    category_dd.options = ['All'] + sorted(data[data['Region'] == region_dd.value]['Category'].unique()) if region_dd.value != 'All' else ['All']
    update_sub_category()

def update_sub_category(*args):
    sub_category_dd.options = ['All'] + sorted(data[(data['Region'] == region_dd.value) & (data['Category'] == category_dd.value)]['Sub-Category'].unique()) if category_dd.value != 'All' else ['All']
    update_product_name()

def update_product_name(*args):
    product_name_dd.options = ['All'] + sorted(data[(data['Region'] == region_dd.value) & (data['Category'] == category_dd.value) & (data['Sub-Category'] == sub_category_dd.value)]['Product Name'].unique()) if sub_category_dd.value != 'All' else ['All']

# Define the analytics functions (the same as in the previous examples)
# ...

# Exit Button Implementation
def exit_program(b):
    with output:
        clear_output()
        print("Exiting the Sales Data Exploration Tool. Thank you for using it!")

# Analytics selection function (the same as in the previous examples)
# ...

# Observers
region_dd.observe(update_category, 'value')
category_dd.observe(update_sub_category, 'value')
sub_category_dd.observe(update_product_name, 'value')
analytics_dd.observe(lambda change: run_analytics(change.new), names='value')
exit_button.on_click(exit_program)

# Initialize dropdowns
update_category()
update_sub_category()
update_product_name()

# Display Components of menu
print("Welcome to the Tableau Sales Data Exploration Tool!")
print("Navigate through the data using the dropdown menus below.")
print("Instructions:")
print("1. Select a Region to start your exploration.")
print("2. Choose a Category, Sub-Category, and Product based on your Region selection.")
print("3. Choose an analytic to view the data.")
print("4. To exit, click the 'Exit' button.")
display(region_dd, category_dd, sub_category_dd, product_name_dd, analytics_dd, exit_button, output)

# Initialize with a default view
run_analytics(analytics_dd.value)
