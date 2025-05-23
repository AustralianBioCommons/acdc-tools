"""Test module for the write_data module."""
import json
import os
import shutil
import tempfile
import time
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from acdctools.validation_reporter.write_data import WriteData

# Sample test data for property definition CSV
PROP_DEF_CSV = """VARIABLE_NAME,OBJECT
patient_id,subject
age,subject
gender,subject
tumor_size,tumor_sample
sample_type,tumor_sample
"""

# Sample test data for link definition CSV
LINK_DEF_CSV = """SCHEMA,PARENT
tumor_sample,subject
"""

# Sample input DataFrame
def create_sample_dataframe():
    """Create a sample input DataFrame for testing."""
    return pd.DataFrame({
        'patient_id': ['PAT001', 'PAT002'],
        'age': [45, 60],
        'gender': ['M', 'F'],
        'tumor_size': [2.5, 3.1],
        'sample_type': ['primary', 'metastasis']
    })

@pytest.fixture
def setup_test_environment():
    """Initialize the test environment with a WriteData instance and test data.

    Returns:
        tuple: (writer, temp_dir, study_id)
    """
    # Create temporary directory for test files
    temp_dir = tempfile.mkdtemp()

    # Create property definition CSV
    prop_def_path = os.path.join(temp_dir, 'prop_def.csv')
    with open(prop_def_path, 'w') as f:
        f.write(PROP_DEF_CSV)

    # Create link definition CSV
    link_def_path = os.path.join(temp_dir, 'link_def.csv')
    with open(link_def_path, 'w') as f:
        f.write(LINK_DEF_CSV)

    # Create sample input dataframe
    input_df = create_sample_dataframe()

    # Create WriteData instance
    study_id = "TEST_STUDY"
    writer = WriteData(
        input_df=input_df,
        prop_def_csv=prop_def_path,
        study_id=study_id,
        link_def_csv_path=link_def_path
    )
    node_indexes = writer.node_indexes
    if 'None' in node_indexes:
        del node_indexes['None']
    writer.node_indexes = node_indexes

    yield writer, temp_dir, study_id

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)

def test_initialization(setup_test_environment):
    """Test initialization of WriteData class."""
    writer, _, study_id = setup_test_environment

    # Test basic initialization
    assert writer is not None
    assert writer.study_id == study_id
    # Should have 2 nodes: subject and tumor_sample
    assert len(writer.node_indexes) == 2
    assert 'subject' in writer.node_indexes
    assert 'tumor_sample' in writer.node_indexes

def test_what_object_node(setup_test_environment):
    """Test the what_object_node method."""
    writer, _, _ = setup_test_environment

    # Test existing variables
    assert writer.what_object_node('patient_id') == 'subject'
    assert writer.what_object_node('tumor_size') == 'tumor_sample'

    # Test non-existing variable
    assert writer.what_object_node('non_existent') is None

def test_split_to_nodes(setup_test_environment):
    """Test the split_to_nodes method."""
    writer, _, _ = setup_test_environment

    node_indexes = writer.split_to_nodes()

    # Check if all expected nodes are present
    assert 'subject' in node_indexes
    assert 'tumor_sample' in node_indexes

    # Check if indexes are correctly assigned
    # patient_id, age, gender should be in subject
    assert len(node_indexes['subject']) == 3
    # tumor_size, sample_type should be in tumor_sample
    assert len(node_indexes['tumor_sample']) == 2

def test_apply_linkage_metadata(setup_test_environment):
    """Test the apply_linkage_metadata method."""
    writer, _, study_id = setup_test_environment

    # Test subject node (which has a parent in our test data)
    subject_data = [{
        'patient_id': 'PAT001',
        'age': 45,
        'gender': 'M'
    }]
    # Mock get_backlinks to return ['subject'] for subject
    with patch.object(writer, 'get_backlinks', return_value=['subject']):
        result = writer.apply_linkage_metadata(subject_data, 'subject')

    assert result[0]['type'] == 'subject'
    assert result[0]['submitter_id'] == f"{study_id}_subject_PAT001"
    assert 'patient_id' in result[0]  # patient_id should remain for subject
    assert 'subjects' in result[0]  # Should have backreference to subject

    # Test tumor_sample node (which has a parent in our test data)
    sample_data = [{
        'patient_id': 'PAT001',
        'tumor_size': 2.5,
        'sample_type': 'primary'
    }]
    # Mock get_backlinks to return ['subject'] for tumor_sample
    with patch.object(writer, 'get_backlinks', return_value=['subject']):
        result = writer.apply_linkage_metadata(sample_data, 'tumor_sample')

    assert result[0]['type'] == 'tumor_sample'
    assert result[0]['submitter_id'] == f"{study_id}_tumor_sample_PAT001"
    # patient_id should be removed for non-subject nodes
    assert 'patient_id' not in result[0]
    # Should have backreference to subject
    assert 'subjects' in result[0]
    assert result[0]['subjects']['submitter_id'] == f"{study_id}_subject_PAT001"

def test_write_nodes_to_json(setup_test_environment):
    """Test the write_nodes_to_json method."""
    writer, temp_dir, study_id = setup_test_environment

    # Create output directory
    output_dir = os.path.join(temp_dir, 'output')
    writer.output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Mock the apply_linkage_metadata method to avoid dependency on get_backlinks
    def mock_apply_linkage_metadata(input_array, node_type):
        for item in input_array:
            item['type'] = node_type
            item['submitter_id'] = f"{study_id}_{node_type}_PAT001"
            if node_type != 'subject':
                item['subjects'] = {'submitter_id': f"{study_id}_subject_PAT001"}
        return input_array

    # Create a copy of node_indexes without the 'None' key for testing
    test_node_indexes = {k: v for k, v in writer.node_indexes.items() if k != 'None'}
    with patch.object(writer, 'node_indexes', test_node_indexes), \
         patch.object(writer, 'apply_linkage_metadata', side_effect=mock_apply_linkage_metadata):
        writer.write_nodes_to_json()

    # Check if files were created
    subject_file = os.path.join(output_dir, 'subject.json')
    tumor_sample_file = os.path.join(output_dir, 'tumor_sample.json')

    assert os.path.exists(subject_file)
    assert os.path.exists(tumor_sample_file)

    # Check subject file content
    with open(subject_file, 'r', encoding='utf-8') as f:
        subject_data = json.load(f)
        assert len(subject_data) == 2  # Two patients
        assert subject_data[0]['type'] == 'subject'
        assert subject_data[0]['submitter_id'] == f"{study_id}_subject_PAT001"

    # Check tumor_sample file content
    with open(tumor_sample_file, 'r', encoding='utf-8') as f:
        sample_data = json.load(f)
        assert len(sample_data) == 2  # Two samples
        assert sample_data[0]['type'] == 'tumor_sample'
        # Should have backreference to subject
        assert 'subjects' in sample_data[0]

def test_create_output_dir(setup_test_environment, monkeypatch):
    """Test the create_output_dir method."""
    writer, temp_dir, _ = setup_test_environment

    # Mock time to have a predictable output directory name
    mock_time = MagicMock(return_value=1640995200)  # 2022-01-01 00:00:00
    mock_localtime = MagicMock(return_value=time.struct_time((2022, 1, 1, 0, 0, 0, 0, 0, 0)))
    
    monkeypatch.setattr(time, 'time', mock_time)
    monkeypatch.setattr(time, 'localtime', mock_localtime)

    # Test directory creation
    save_dir = os.path.join(temp_dir, 'test_output')
    received_date = '20220101'

    writer.create_output_dir(
        study_id='TEST_STUDY',
        received_date=received_date,
        save_dir=save_dir
    )

    # Check if directory was created with correct naming pattern
    expected_path = os.path.join(
        save_dir,
        f"{received_date}_TEST_STUDY_cleaned/2022-01-01_0000"
    )
    assert os.path.exists(expected_path)
    assert writer.output_dir == expected_path
