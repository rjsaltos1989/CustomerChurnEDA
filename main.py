#----------------------------------------------------------------------------------
# Unsupervised Churn Prediction
#----------------------------------------------------------------------------------
# Implementation: Ramiro Saltos
# Date: 2025-08-30
# Version: 1.0
#----------------------------------------------------------------------------------

# Libraries
#----------------------------------------------------------------------------------
from utils import *

#----------------------------------------------------------------------------------
# Data Preparation
#----------------------------------------------------------------------------------

# Import the dataset
df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')
df = pd.concat([df_train, df_test])

# Show the first rows of the dataset
print(df.head().T)

# Show the dtypes of the columns to fix if necessary
print(df.dtypes)

# Standardize columns names: lowercase and replace spaces with underscores.
df.columns = df.columns.str.lower().str.replace(' ', '_')

# Remove customerid from data
df = df.drop('customerid', axis=1)

# Make churn a categorical variable
df['churn'] = ["churn" if val == 1 else "no_churn" for val in df['churn']]

# Select the string columns in the dataframe. The dtype 'object' indicates that the column contains strings.
string_columns = df.select_dtypes(include=['object', 'string']).columns

# Select the numerical columns in the dataframe.
num_cols = df.select_dtypes(include=['number']).columns

# Lowercases and replaces spaces with underscores for values in all string columns.
for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')

# Look for missing values
print(df.isnull().sum())

# Show the rows with missing values
print(df[df.isna().any(axis=1)])

# Drop the single row with missing values
df = df.dropna()

# %% Univariate Analysis
#----------------------------------------------------------------------------------

# Make barplots for the categorical variables
make_barplot(df, string_columns)

# Make histograms and box plots for the numerical variables
for num_var in num_cols:
    make_histogram(df, num_var)
    make_boxplot(df, num_var)

# Compute descriptive statistics for the numerical variables
num_summary = df[num_cols].describe().T
num_summary.reset_index(inplace=True)
num_summary.rename(columns={'index': 'variable'}, inplace=True)

# %% Multivariate Analysis
#---------------------------------------------------------------------------------

# Make stacked barplots for the categorical variables
make_stacked_barplots(df, string_columns)

# Make stacked boxplots for the categorical and numerical variables
make_grouped_boxplots(df, num_cols, string_columns)

# Compute correlation matrix for the numerical variables
make_heat_map(df, num_cols)

# Compute the scatterplot matrix for the numerical variables
make_scatter_matrix(df, num_cols)



















