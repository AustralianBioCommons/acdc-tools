import time
import pandas as pd
import os
import json
class WriteData:

    def __init__(self, input_df, prop_def_csv: str, study_id: str, link_def_csv_path: str):
        """
        Initialize the write_data class with input dataframe and property definition CSV.

        Parameters:
        input_df : DataFrame
            The input dataframe containing the data to be processed.
        prop_def_csv : str
            The file path to the CSV containing property definitions that match to the data node.
        """
        self.output_dir = None
        self.input_df = input_df
        self.study_id = study_id
        self.prop_def = pd.read_csv(prop_def_csv)
        self.node_indexes = self.split_to_nodes()
        self.link_def_df = self.load_link_def_csv(link_def_csv_path)
        
    def read_csv(self, file_path):
        dataframe = pd.read_csv(file_path)
        return dataframe


    def create_output_dir(self, study_id, received_date, save_dir):
        created_time = time.strftime("%Y-%m-%d_%H%M", time.localtime())
        pre_output_dir = f"{save_dir}/{received_date}_{study_id}_cleaned"
        output_dir = f"{pre_output_dir}/{created_time}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"=== AWS S3 Copy From: {pre_output_dir} ===")
        print(f"Created output dir: {output_dir}")
        self.output_dir = output_dir


    def write_json(self, filename):
        self.input_df.to_json(f"{self.output_dir}/{filename}.json", orient='records')
        print(f"Dataframe successfully written to {self.output_dir}/{filename}.json")
        
        
    def what_object_node(self, var_name):
        if var_name in self.prop_def['VARIABLE_NAME'].values:
            return self.prop_def.loc[self.prop_def['VARIABLE_NAME'] == var_name, 'OBJECT'].iloc[0]
        return None
        
        
    def split_to_nodes(self):
        colnames = list(self.input_df.columns)
        node_list = []
        for col in colnames:
            node_list.append(self.what_object_node(col))
        
        unqiue_node_names = set(node_list)
        
        output = {}
        for node in unqiue_node_names:
            indexes = [index for index, value in enumerate(node_list) if value == node]
            node_str = str(node)
            output[node_str] = indexes
            
        return output
    
    def load_link_def_csv(self, link_def_csv_path: str):
        link_def_df = pd.read_csv(link_def_csv_path)
        return link_def_df
    
    def get_backlinks(self, input_node):
        df = self.link_def_df
        parent_nodes = df[df['SCHEMA'] == input_node]['PARENT'].tolist()
        return parent_nodes
        
    
    def apply_linkage_metadata(self, input_array: list, node_type: str):
        
        parent_node = self.get_backlinks(node_type)[0]
        
        for i, obj in enumerate(input_array):
            obj['type'] = node_type
            obj['submitter_id'] = f"{self.study_id}_{obj['type']}_{obj['patient_id']}"
            node_ref = f"{parent_node}s"
            
            if node_type == "subject":
                backref_submitter_id = f"{self.study_id}"             
            else:
                backref_submitter_id = f"{self.study_id}_{parent_node}_{obj['patient_id']}"
                del obj['patient_id']
    
            obj[node_ref] = {'submitter_id': backref_submitter_id}
            
            
        return input_array
    
    def write_nodes_to_json(self):
        # remove indexes for colnames not found in schema
        node_indexes = self.node_indexes
        del node_indexes['None']
        
        # this loop subsets the input dataframe based on the indexes for each node
        for node, value in self.node_indexes.items():
            
            # patient_id with index 0 should be added to every node reference
            if 0 in value:
                value.remove(0)
            value.append(0)
            
            # print(value)
            output_df = self.input_df.iloc[:, value]
            output_file_path = f"{self.output_dir}/{node}.json"

            output_json = json.loads(output_df.to_json(orient='records'))
            output_json = self.apply_linkage_metadata(output_json, node)
            
            with open(output_file_path, 'w') as f:
                json.dump(output_json, f, indent=4)
                
            print(f"Dataframe for node '{node}' successfully written to {output_file_path}")