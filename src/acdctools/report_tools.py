import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from io import BytesIO, StringIO
import base64
import json
import numpy as np
import os
import sys
import importlib
import pprint
import pickle
import papermill as pm
import subprocess
import shutil



class ParseData:
    def __init__(self, file_path):
        """
        __init__ initializes a ParseCSV object.

        Parameters
        ----------
        filename : str
            The name of the CSV file to read into the ParseCSV object.
        dir : str, optional
            The directory containing the data file..
        """
        self.filename = file_path
    
    def load_csv(self, parse_dates=True):
        
        if not self.filename.endswith('.csv'):
            raise ValueError("File is not .csv file")

        self.df = pd.read_csv(f"{self.filename}", parse_dates=parse_dates)
        print(f"Successfully Loaded: {self.filename}")

    def load_xlsx(self, parse_dates=True):

        if not self.filename.endswith('.xlsx'):
            raise ValueError("The file is not an .xlsx Excel file.")
        
        self.df = pd.read_excel(self.filename, parse_dates=parse_dates)
        print(f"Successfully Loaded: {self.filename}")
        
    def format_dates(self, date_columns, date_format='%d/%m/%Y'):
        """
        Format the specified date columns in the dataframe to 'yyyy-mm-dd'.

        Parameters
        ----------
        date_columns : list
            A list of column names in the dataframe that contain date values to be formatted.
        date_format : str, optional
            The format of the date in the input data. Default is 'dd/mm/yyyy'.
        
        Returns
        -------
        None
        """
        for col in date_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(
                    lambda x: pd.to_datetime(x, format=date_format, errors='coerce').strftime('%Y-%m-%d') 
                    if pd.notnull(x) else x
                )
            else:
                print(f"Warning: Column '{col}' not found in the dataframe.")
                
    def convert_columns_to_string(self, columns):
        """
        Convert the specified columns in the dataframe to string type.

        Parameters
        ----------
        columns : list
            A list of column names in the dataframe to be converted to string type.
        
        Returns
        -------
        None
        """
        for col in columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str)
            else:
                print(f"Warning: Column '{col}' not found in the dataframe.")
                
    def return_df(self):
        return self.df
    
    def convert_columns_to_int(self, columns, fill_value=0):
        """
        Convert the specified columns in the dataframe to integer type, handling NaN values.

        Parameters
        ----------
        columns : list
            A list of column names in the dataframe to be converted to integer type.
        fill_value : int, optional
            The value to replace NaN values with before converting to integer. Default is 0.
        
        Returns
        -------
        None
        """
        for col in columns:
            if col in self.df.columns:
                try:
                    # Replace NaN values with the specified fill value before converting
                    self.df[col] = self.df[col].fillna(fill_value).astype(int)
                except ValueError as e:
                    print(f"Error: Could not convert column '{col}' to int. {e}")
            else:
                print(f"Warning: Column '{col}' not found in the dataframe.")

class DataSummary:
    def __init__(self, dataframe, dir="data"):
        """
        __init__ initializes a DataSummary object.

        Parameters
        ----------
        dataframe : DataFrame
            The dataframe to be summarized.
        dir : str, optional
            The directory containing the data file.
        """
        
        self.dir = dir
        self.df = dataframe

    def dataframe_to_base64_csv(self, dataframe):
        csv_buffer = StringIO()
        dataframe.to_csv(csv_buffer, index=False)
        csv_str = csv_buffer.getvalue()
        b64_csv = base64.b64encode(csv_str.encode()).decode()
        return b64_csv

    def get_data_types(self, export_base64=False):
        data_types = self.df.dtypes.to_frame(name='dtype').reset_index().rename(columns={'index': 'variable'})
        if export_base64:
            return self.dataframe_to_base64_csv(data_types)
        return data_types

    def basic_numeric_stats(self, export_base64=False):
        summary = self.df.describe()
        summary_pivot = summary.reset_index().melt(id_vars='index')
        summary_pivot['variable'] = pd.Categorical(summary_pivot['variable'], categories=self.df.columns, ordered=True)
        summary_pivot = summary_pivot.pivot(index='variable', columns='index', values='value')

        summary_pivot['count'] = summary_pivot["count"].round(0).astype(int)
        summary_pivot['max'] = summary_pivot["max"].round(1)
        summary_pivot['mean'] = summary_pivot["mean"].round(1)
        summary_pivot['min'] = summary_pivot["min"].round(1)
        summary_pivot['std'] = summary_pivot["std"].round(1)

        summary_pivot = summary_pivot.loc[:, ~summary_pivot.columns.str.contains('%')].reset_index()
        summary_pivot.index.name = None  # Ensure index name is not set
        summary_pivot.columns.name = 'index'  # Set the columns name to 'index'

        data_types = self.get_data_types()
        summary_pivot = summary_pivot.merge(data_types, on='variable', how='left')
        summary_pivot = summary_pivot[['variable', 'dtype'] + [col for col in summary_pivot.columns if col not in ['variable', 'dtype']]]

        if export_base64:
            return self.dataframe_to_base64_csv(summary_pivot)
        return summary_pivot

    def mis_dist_stats(self, export_base64=False):
        miss = self.df.isnull().sum().to_frame(name='missing')
        dist = self.df.nunique().to_frame(name='distinct')
        combined = miss.join(dist)
        combined['variable'] = pd.Categorical(combined.index, categories=self.df.columns, ordered=True)
        combined = combined[['variable', 'missing', 'distinct']]
        combined = combined.reset_index(drop=True)
        combined.index.name = None  # Ensure index name is not set
        combined.columns.name = 'index'  # Set the columns name to 'index'

        data_types = self.get_data_types()
        combined = combined.merge(data_types, on='variable', how='left')
        combined = combined[['variable', 'dtype'] + [col for col in combined.columns if col not in ['variable', 'dtype']]]

        if export_base64:
            return self.dataframe_to_base64_csv(combined)
        return combined

    def enum_counts(self, export_base64=False):
        cat_vars = self.df.select_dtypes(include=['object']).columns

        def enum_table_summary(dataframe):
            summary = dataframe.value_counts().reset_index()
            summary.columns = ['Value', 'Count']
            return summary

        output_tables = []
        for cat_var in cat_vars:
            out = enum_table_summary(self.df.loc[:, cat_var])
            output_tables.append((cat_var, out))

        if export_base64:
            return [(cat_var, self.dataframe_to_base64_csv(out)) for cat_var, out in output_tables]
        return output_tables

    def combined_numeric_stats(self, export_base64=False):
        numeric_summary = self.basic_numeric_stats()
        missing = self.mis_dist_stats()
        distinct = self.mis_dist_stats()

        combined_summary = pd.concat([
            numeric_summary, 
            missing, 
            distinct
        ], axis=1)

        if export_base64:
            return self.dataframe_to_base64_csv(combined_summary)
        return combined_summary

    def plot_density(self):
        numeric_columns = self.df.select_dtypes(include=['number']).columns

        # Create a grid for the plots
        num_columns = 4
        num_rows = (len(numeric_columns) + num_columns - 1) // num_columns
        fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, num_rows * 3))
        axes = axes.flatten()

        for i, column in enumerate(numeric_columns):
            if not self.df[column].isnull().all():
                sns.kdeplot(self.df[column].dropna(), fill=True, ax=axes[i])
                axes[i].set_title(f'{column}')
                axes[i].set_xlabel(column)
                axes[i].set_ylabel('Density')
                axes[i].grid(True)
            else:
                axes[i].set_visible(False)

        plt.tight_layout()

        # Convert plot to base64
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        return img_base64


def df_to_base64(df):
    """
    Convert a DataFrame to a base64 encoded CSV string.

    Parameters:
    - df (pd.DataFrame): Data frame to convert.

    Returns:
    - str: Base64 encoded CSV string.
    """
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def string_to_base64(input_string):
    """
    Convert a string to a base64 encoded string.

    Parameters:
    - input_string (str): String to convert.

    Returns:
    - str: Base64 encoded string.
    """
    return base64.b64encode(input_string.encode('utf-8')).decode('utf-8')

def list_to_base64(input_list):
    """
    Convert a list to a base64 encoded JSON string.

    Parameters:
    - input_list (list): List to convert.

    Returns:
    - str: Base64 encoded JSON string.
    """
    json_string = json.dumps(input_list)
    return base64.b64encode(json_string.encode('utf-8')).decode('utf-8')

def get_colnames(df):
    """
    Get column names of a DataFrame.

    Parameters:
    - df (pd.DataFrame): Data frame to get column names from.

    Returns:
    - list: List of column names.
    """
    return list(df.columns)

def get_schema_variables(schema_path, nodes, drop_nodes=None, exclude_schema_vars=None):
    """
    Get variable names from a schema.

    Parameters:
    - schema_path (str): Path to the schema file.
    - nodes (list): List of nodes to extract variables from.
    - drop_nodes (list, optional): List of nodes to drop. Default is None.
    - exclude_schema_vars (list, optional): List of variables to exclude. Default is None.

    Returns:
    - list: List of variable names.
    """
    if drop_nodes is None:
        drop_nodes = ['$ref', 'consent_codes', 'subjects', 'projects']
    if exclude_schema_vars is None:
        exclude_schema_vars = []

    with open(schema_path) as f:
        schema = json.load(f)

    variables = []
    for node in nodes:
        vars = schema[f"{node}.yaml"]['properties'].keys()
        variables.extend(vars)

    variables = list(set(variables))  # Remove duplicates
    variables = [var for var in variables if var not in drop_nodes]
    variables = [var for var in variables if var not in exclude_schema_vars]

    return variables

def compare_vars(schema_vars, data_vars):
    """
    Compare data column names and schema variable names, report any differences.

    Parameters:
    - schema_vars (list): List of schema variables.
    - data_vars (list): List of data variables.

    Returns:
    - tuple: Found, missing, and unassigned variables.
    """
    found = [var for var in schema_vars if var in data_vars]
    missing = [var for var in schema_vars if var not in data_vars]
    unassigned = [var for var in data_vars if var not in schema_vars]
    return found, missing, unassigned

def variable_summary(schema_vars, data_vars):
    """
    Summarize variable comparison results.

    Parameters:
    - schema_vars (list): List of schema variables.
    - data_vars (list): List of data variables.

    Returns:
    - dict: Summary of variable comparison.
    """
    found, missing, unassigned = compare_vars(schema_vars, data_vars)
    n_total = len(schema_vars)
    return {
        "fraction_found": f"{len(found)}/{n_total}",
        "percentage_found": round(100 * (len(found) / n_total), 2),
        "found_list": found,
        "missing_list": missing,
        "unassigned_list": unassigned
    }

def test_duplicate_ids(df, id_colname='id'):
    """
    Test for duplicate IDs in a DataFrame.

    Parameters:
    - df (pd.DataFrame): Data frame to test.
    - id_colname (str): Column name of the ID.

    Returns:
    - dict: Number of duplicates and list of duplicated IDs.
    """
    dup_bool = df.duplicated(subset=[id_colname])
    return {
        'n_duplicated': dup_bool.sum(),
        'duplicated_ids': df[dup_bool][id_colname].tolist()
    }

def find_categorical_var_frequencies(df, exclude_columns=None, export_base64=False):
    """
    Report the frequency of all values in categorical variables as a percentage, with an option to exclude certain columns.

    Parameters:
    - df (pd.DataFrame): The dataframe containing the data.
    - exclude_columns (list, optional): A list of column names to exclude from the frequency report. Default is None.
    - export_base64 (bool, optional): If True, returns the DataFrame as a base64 encoded CSV string. Default is False.

    Returns:
    - pd.DataFrame or str: A DataFrame with columns 'variable', 'value', and 'frequency' (as a percentage) for all values in categorical variables, sorted by variable and frequency.
                            If export_base64 is True, returns a base64 encoded CSV string instead.
    """
    if exclude_columns is None:
        exclude_columns = []

    frequency_data = []
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if col not in exclude_columns:
            value_counts = df[col].value_counts(normalize=True) * 100
            for value, freq in value_counts.items():
                frequency_data.append({'variable': col, 'value': value, 'frequency': round(freq, 2)})
    
    frequency_df = pd.DataFrame(frequency_data)
    frequency_df = frequency_df.sort_values(by=['variable', 'frequency'], ascending=[True, False])
    
    if export_base64:
        buffer = BytesIO()
        frequency_df.to_csv(buffer, index=False)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')
    
    return frequency_df

def plot_numeric_distributions_with_outliers(df, n_cols=4, lower_percentile=1, upper_percentile=99, plot_width=5, plot_height=6):
    """
    Plots the distribution of numeric variables in the dataframe using violin plots and highlights the outliers.

    Parameters:
    - df (pd.DataFrame): The dataframe containing the data.
    - n_cols (int): Number of columns in the grid layout for the plots. Default is 4.
    - lower_percentile (float): Lower percentile to determine the lower bound for outliers. Default is 1.
    - upper_percentile (float): Upper percentile to determine the upper bound for outliers. Default is 99.
    - plot_width (int): Width of each individual plot. Default is 5.
    - plot_height (int): Height of each individual plot. Default is 6.

    Returns:
    - str: Base64 encoded string of the plot.
    """
    numeric_cols = df.select_dtypes(include=['number']).columns
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    sns.set(style="whitegrid")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(plot_width * n_cols, plot_height * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        sns.violinplot(data=df, y=col, ax=axes[i], inner=None, color="skyblue", linewidth=0, alpha=0.7)
        
        # Calculate the lower and upper bounds using Tukey's fence method
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Highlight outliers
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        sns.stripplot(data=outliers, y=col, ax=axes[i], color="lightpink", size=4, jitter=False)
        
        # Add median dot
        median_value = df[col].median()
        axes[i].scatter([0], [median_value], color='grey', zorder=5, s=50, label='Median')
        
        axes[i].set_ylabel(col, fontsize=12)
        axes[i].set_xlabel('')
        axes[i].tick_params(axis='y', labelsize=10)
        axes[i].tick_params(axis='x', labelsize=10)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    # Convert plot to base64
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return img_base64

def plot_missing_values_percentage(df, figsize=(12, 8), xlabel_fontsize=12, ylabel_fontsize=12, title_fontsize=15):
    """
    Plots a bar graph showing the percentage of missing values for each variable in the dataframe.

    Parameters:
    - df (pd.DataFrame): The dataframe containing the data.
    - figsize (tuple): Size of the figure (width, height). Default is (12, 8).
    - xlabel_fontsize (int): Font size for the x-axis label. Default is 12.
    - ylabel_fontsize (int): Font size for the y-axis label. Default is 12.
    - title_fontsize (int): Font size for the title. Default is 15.

    Returns:
    - str: Base64 encoded string of the plot.
    """
    # Calculate the percentage of missing values for each column
    missing_percentage = df.isnull().mean() * 100

    # Sort the missing percentage in descending order
    missing_percentage = missing_percentage.sort_values(ascending=False)

    # Plot the bar graph
    plt.figure(figsize=figsize)
    sns.barplot(x=missing_percentage.index, y=missing_percentage, hue=missing_percentage.index, dodge=False, palette="viridis", legend=False)
    plt.xticks(rotation=90)
    plt.xlabel('Variables', fontsize=xlabel_fontsize)
    plt.ylabel('Percentage of Missing Values', fontsize=ylabel_fontsize)
    plt.title('Percentage of Missing Values by Variable', fontsize=title_fontsize)

    # Adjust layout to ensure the bottom of the graph is not cut off
    plt.tight_layout()

    # Convert plot to base64
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return img_base64

def tag_unresolvable(df, rows):
    """
    Tag specified rows as unresolvable.

    Parameters:
    - df (pd.DataFrame): Data frame to tag.
    - rows (list): List of row indices to tag as unresolvable.

    Returns:
    - pd.DataFrame: Data frame with tagged rows.
    """
    for row in rows:
        df.at[row, 'unresolvable'] = 'unresolvable'
    return df

def extract_error_messages(strings, pattern='is '):
    return [s.split(pattern, 1)[1] if pattern in s else s for s in strings]

def collapse_validation_errors(df):
    df['validation_message'] = extract_error_messages(df['validation_error'])
    collapsed_df = df.groupby(['invalid_key', 'validation_message']).size().reset_index(name='error_count')
    return collapsed_df

class ValidationStats:
    def __init__(self, input_df, schema_path, resolved_schema_path, categorical_vars: list = [], 
                 nodes: list = ["subject", "demographic", "exposure", "lab_result", "medical_history", "medication", "blood_pressure_test"], 
                 exclude_schema_vars: list = ['age_at_collection', 'collection_stage', 'baseline_timepoint', 'alternate_timepoint']):
        """
        Initializes a ValidationReport object.

        Parameters:
        - input_df (pd.DataFrame): Input DataFrame to be validated.
        - schema_path (str): Path to the schema file to validate against.
        - resolved_schema_path (str): Path to resolved schema created using the gen3schemadev.gen3validate.SchemaResolver class.
        - categorical_vars (list): List of categorical variables.
        - nodes (list): List of nodes to include in the schema variables.
        - exclude_schema_vars (list): List of schema variables to exclude.

        Returns:
        - ValidationReport: A ValidationReport object containing the summary and validation results.
        """
        print("Initializing ValidationStats")
        self.df = input_df
        self.schema_path = schema_path
        self.resolved_schema_path = resolved_schema_path
        self.cat_vars = categorical_vars
        self.nodes = nodes
        self.exclude_schema_vars = exclude_schema_vars
        self.create_summary()
        
    def create_summary(self):
        print("Creating summary")
        self.data_summary = DataSummary(self.df)
        self.numeric_stats_df = self.data_summary.basic_numeric_stats()
        print("Numeric stats dataframe created")
        self.mis_dist_df = self.data_summary.mis_dist_stats()
        print("Missing distribution dataframe created")
        self.enum_counts_df = self.data_summary.enum_counts()
        print("Enum counts dataframe created")

        self.numeric_stats_base64 = self.data_summary.basic_numeric_stats(export_base64=True)
        print("Numeric stats base64 created")
        self.mis_dist_base64 = self.data_summary.mis_dist_stats(export_base64=True)
        print("Missing distribution base64 created")
        self.enum_counts_base64 = self.data_summary.enum_counts(export_base64=True)
        print("Enum counts base64 created")

        self.schema_vars = get_schema_variables(
            self.schema_path, 
            nodes=self.nodes, 
            exclude_schema_vars=self.exclude_schema_vars
        )
        print("Schema variables obtained")
        self.data_vars = get_colnames(self.df)
        print("Data variables obtained")

        self.var_sum = variable_summary(self.schema_vars, self.data_vars)
        print("Variable summary created")
        self.fraction_found = self.var_sum['fraction_found']
        self.percentage_found = self.var_sum['percentage_found']
        self.found_list = self.var_sum['found_list']
        self.missing_list = self.var_sum['missing_list']
        self.unassigned_list = self.var_sum['unassigned_list']

        self.dup_id = test_duplicate_ids(self.df, 'patient_id')
        print("Duplicate IDs tested")
        self.n_dup_id = self.dup_id['n_duplicated']
        self.dup_id_list = self.dup_id['duplicated_ids']

        self.cat_freq_df = find_categorical_var_frequencies(self.df, self.cat_vars, export_base64=False)
        print("Categorical frequencies dataframe created")
        self.cat_freq_df_64 = find_categorical_var_frequencies(self.df, self.cat_vars, export_base64=True)
        print("Categorical frequencies base64 created")

        self.numeric_outlier_plot = plot_numeric_distributions_with_outliers(self.df, n_cols=8, lower_percentile=0.5, upper_percentile=99.5, plot_width=2)
        print("Numeric outlier plot created")

        self.missing_val_plot = plot_missing_values_percentage(self.df, figsize=(15, 8))
        print("Missing values plot created")

    def create_validation_results(self, validation_results):
        print("Creating validation results")
        self.full_validation_df = validation_results.output.reset_index().drop(columns=['index'])
        print("Full validation dataframe created")
        self.full_validation_df['unresolvable'] = 'resolvable'
        self.full_validation_df_base64 = df_to_base64(self.full_validation_df)
        print("Full validation dataframe base64 created")

    def create_collapsed_validation_results(self):
        print("Creating collapsed validation results")
        self.collapsed_validation_df = self.full_validation_df.copy()
        self.collapsed_validation_df = collapse_validation_errors(self.collapsed_validation_df)
        print("Collapsed validation dataframe created")
        self.collapsed_validation_df = self.collapsed_validation_df[['invalid_key', 'error_count', 'validation_message']]

        self.collapsed_validation_df_base64 = df_to_base64(self.collapsed_validation_df)
        print("Collapsed validation dataframe base64 created")
        self.collapsed_validation_total_errors = np.sum(self.collapsed_validation_df['error_count'])
        print("Total errors in collapsed validation dataframe calculated")
    
    def check_validation_df(self):
        print("Checking validation dataframe")
        return self.full_validation_df
    
    def filter_missing_values(self, column_name='input_value'):
        # Replace 'None' and 'NaN' strings with actual NaN values in the specified column
        self.full_validation_df.replace({column_name: ['None', 'NaN', 'nan']}, np.nan, inplace=True)
        # Drop rows where the specified column has NaN values
        self.full_validation_df.dropna(subset=[column_name], inplace=True)
        print("Filtered missing values")
        self.full_validation_df_base64 = df_to_base64(self.full_validation_df)
        print("Full validation dataframe base64 created")
        
        
    def filter_unresolvable(self, unresolvable_rows, desired_order: list = ['row', 'invalid_key', 'validation_error', 'unresolvable', 'input_value', 'validator_value']):
        print("Filtering validation results")
        self.full_validation_df = tag_unresolvable(self.full_validation_df, unresolvable_rows)
        print("Tagged unresolvable rows")
        # Reorder columns of collapsed_validation_df
        self.full_validation_df = self.full_validation_df[desired_order]
        print("Reordered columns of validation dataframe")
        # Filter unresolvable rows
        self.filtered_validation_df = self.full_validation_df[self.full_validation_df['unresolvable'] == 'unresolvable']
        print("Filtered unresolvable rows")
        self.filtered_validation_df_base64 = df_to_base64(self.filtered_validation_df)
        print("Filtered validation dataframe base64 created")
        self.full_validation_df_base64 = df_to_base64(self.full_validation_df)
        print("Full validation dataframe base64 created")
    



class ValidationReportCompiler:
    def __init__(self, study_id: str, base_save_dir: str = '/output/qc_pickle', output_dir: str = None, parameters: dict = None):
        self.study_id = study_id
        self.base_save_dir = base_save_dir
        self.output_dir = output_dir
        self.parameters = parameters
        self.pickle_path = self._generate_pickle_save_path()
        self.execution_path = self._generate_execution_path()

    def _generate_pickle_save_path(self) -> str:
        """
        Generate the file path for saving the pickle file.

        Returns
        -------
        str
            The absolute path for the pickle file.
        """
        received_date = datetime.now().strftime('%Y-%m-%d_%H%M')
        print(f"Generating pickle save path for received_date: {received_date}, study_id: {self.study_id}")
        abs_dir = os.path.abspath('.')
        pickle_path = os.path.join(abs_dir, self.base_save_dir, f"{received_date}_{self.study_id}.pkl")
        print(f"Pickle save path: {pickle_path}")
        return pickle_path

    def save_validation_stats(self, validation_stats: dict) -> None:
        """
        Save the validation statistics to a pickle file.

        Parameters
        ----------
        validation_stats : dict
            The validation statistics to be saved.

        Returns
        -------
        None
        """
        pickle_save_path = self._generate_pickle_save_path()
        print(f"Saving validation stats to {pickle_save_path}")
        pickle_base_dir = os.path.dirname(pickle_save_path)
        print(f"Base directory: {pickle_base_dir}")
        os.makedirs(pickle_base_dir, exist_ok=True)
        with open(pickle_save_path, 'wb') as f:
            pickle.dump(validation_stats, f)
        print("Validation stats saved successfully")

    def _generate_execution_path(self) -> str:
        abs_dir = os.path.abspath('.')
        output_path = f"{abs_dir}/{self.output_dir}/{datetime.now().strftime('%Y-%m-%d_%H%M')}_{self.parameters['study_id']}_validation_report.ipynb"
        return output_path

    def execute_validation_notebook(self, output_path: str = None, template_fn: str = 'templates/template_validation.ipynb') -> str:
        """
        Execute the validation notebook and save it to a file.

        Returns
        -------
        str
            The path to the saved notebook.
        """
        if output_path is None:
            output_path = self.execution_path
            
        pickle_save_path = self._generate_pickle_save_path()
        self.parameters['pickle_path'] = pickle_save_path

        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Output directory: {self.output_dir}")
        print(f"Current dir {os.getcwd()}")
        print(f"Loading pickle from: {pickle_save_path}")
        
        pm.execute_notebook(
            template_fn,
            output_path,
            parameters=self.parameters
        )
        print(f"Validation notebook saved to: {output_path}")

    def pretty_jupyter_to_html(self, notebook_path: str = None) -> None:
        """
        Convert a Jupyter notebook to HTML using a subprocess.

        Parameters
        ----------
        notebook_path : str
            The path to the Jupyter notebook to be converted.

        Returns
        -------
        None
        """
        if notebook_path is None:
            notebook_path = self.execution_path

        # Check if jupyter is available in the environment
        if not shutil.which("jupyter"):
            print("Error: jupyter command not found. Please ensure Jupyter is installed and in your PATH.")
            return

        # Check if the notebook exists
        if not os.path.isfile(notebook_path):
            print(f"Error: Notebook file {notebook_path} does not exist.")
            return

        try:
            command = [
                "jupyter", "nbconvert", "--to", "html", "--execute", "--template", "pj", notebook_path
            ]
            print(f"Converting notebook to HTML: {command}")
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"Notebook converted to HTML successfully: {result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while converting notebook to HTML: {e.stderr}")
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")

