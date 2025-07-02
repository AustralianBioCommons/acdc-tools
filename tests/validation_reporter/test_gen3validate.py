import os
import json
import tempfile
import shutil
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

# Import the module to be tested
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

from acdctools.validation_reporter.gen3validate import (
    SchemaResolver,
    SchemaValidatorSynth,
    QuickValidateSynth,
    SyntheticDataCombiner,
    SchemaValidatorDataFrame,
    ValidationReporter
)

# Test data
SAMPLE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0}
    },
    "required": ["name"]
}

SAMPLE_VALID_DATA = [
    {"name": "John", "age": 30},
    {"name": "Alice", "age": 25}
]

SAMPLE_INVALID_DATA = [
    {"name": "Bob", "age": -5},  # Invalid age
    {"age": 30},  # Missing required name
]

# Fixtures
@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def test_schema_file(temp_dir):
    """Create a test schema file."""
    schema_path = os.path.join(temp_dir, "test_schema.json")
    with open(schema_path, 'w') as f:
        json.dump(SAMPLE_SCHEMA, f)
    return schema_path

@pytest.fixture
def test_data_file(temp_dir):
    """Create a test data file."""
    data_path = os.path.join(temp_dir, "test_data.json")
    with open(data_path, 'w') as f:
        json.dump(SAMPLE_VALID_DATA, f)
    return data_path

# Test SchemaValidatorSynth
class TestSchemaValidatorSynth:
    def test_validation_success(self, test_schema_file):
        """Test successful validation of data against schema."""
        validator = SchemaValidatorSynth(test_schema_file, SAMPLE_VALID_DATA)
        assert validator.results['success_count'] == 2
        assert validator.results['fail_count'] == 0
        assert not validator.errors

    def test_validation_failure(self, test_schema_file):
        """Test validation failure with invalid data."""
        validator = SchemaValidatorSynth(test_schema_file, SAMPLE_INVALID_DATA)
        assert validator.results['fail_count'] > 0
        assert validator.errors

# Test SchemaValidatorDataFrame
class TestSchemaValidatorDataFrame:
    def test_dataframe_validation_success(self, test_schema_file):
        """Test successful validation with DataFrame input."""
        validator = SchemaValidatorDataFrame(SAMPLE_VALID_DATA, test_schema_file)
        results_df, metrics = validator.results, validator.metrics
        
        assert isinstance(results_df, pd.DataFrame)
        assert isinstance(metrics, pd.DataFrame)
        assert (results_df['Validation Result'] == 'SUCCESS').all()
        assert metrics.iloc[0]['success_count'] == len(SAMPLE_VALID_DATA)
        assert metrics.iloc[0]['fail_count'] == 0

# Test ValidationReporter
class TestValidationReporter:
    @patch('acdctools.validation_reporter.gen3validate.ValidationReporter.transform_validate_df')
    @patch('acdctools.validation_reporter.gen3validate.pd.read_csv')
    @patch('acdctools.validation_reporter.gen3validate.json.load')
    @patch('builtins.open')
    def test_validation_reporter_init(self, mock_open, mock_json_load, mock_read_csv, mock_transform):
        """Test ValidationReporter initialization with mocks."""
        # Setup mock data
        mock_schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        mock_json_load.return_value = mock_schema
        
        # Mock file operations
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_open.return_value = mock_file
        
        # Mock DataFrame
        test_df = pd.DataFrame([{"name": "Test"}])
        mock_read_csv.return_value = test_df
        
        # Mock transform
        mock_transform.return_value = pd.DataFrame()
        
        # Create test instance
        reporter = ValidationReporter(schema_path="dummy_schema.json", csv_path="dummy.csv")
        
        # Verify file operations
        mock_open.assert_called_with("dummy_schema.json", 'r')
        mock_json_load.assert_called_with(mock_file)
        mock_read_csv.assert_called_with("dummy.csv", nrows=None)
        
        # Verify attributes
        assert hasattr(reporter, 'data')
        assert hasattr(reporter, 'validator')
        assert hasattr(reporter, 'validate_df')
        assert hasattr(reporter, 'output')

    @patch('acdctools.validation_reporter.gen3validate.ValidationReporter.transform_validate_df')
    @patch('acdctools.validation_reporter.gen3validate.SchemaValidatorDataFrame')
    @patch('acdctools.validation_reporter.gen3validate.json.load')
    @patch('builtins.open')
    def test_validation_reporter_with_dataframe(self, mock_open, mock_json_load, mock_validator, mock_transform):
        """Test ValidationReporter with direct DataFrame input."""
        # Setup mock schema
        mock_schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        mock_json_load.return_value = mock_schema
        
        # Mock file operations
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_open.return_value = mock_file
        
        # Mock validator
        mock_validator_instance = MagicMock()
        mock_validator.return_value = mock_validator_instance
        
        # Setup validation results
        errors_df = pd.DataFrame([{
            'Index': 0,
            'Validation Result': 'PASS',
            'Schema path': '$.name',
            'Invalid key': '',
            'Validation error': '',
            'Validator': 'type',
            'Validator value': 'string'
        }])
        
        mock_validator_instance.validate_schema.return_value = (
            errors_df,
            pd.DataFrame([{'success_count': 1, 'fail_count': 0, 'total': 1}])
        )
        
        # Mock transform
        mock_transform.return_value = errors_df
        
        # Create test data
        test_df = pd.DataFrame([{"name": "Test"}])
        
        # Create test instance with DataFrame
        reporter = ValidationReporter(schema_path="dummy_schema.json", dataframe=test_df)
        
        # Verify data was set correctly
        assert reporter.data == [{"name": "Test"}]
        
        # Verify the transform was called
        mock_transform.assert_called_once()
    
    @patch('acdctools.validation_reporter.gen3validate.ValidationReporter.transform_validate_df')
    @patch('acdctools.validation_reporter.gen3validate.SchemaValidatorDataFrame')
    @patch('acdctools.validation_reporter.gen3validate.json.load')
    @patch('builtins.open')
    def test_validation_process(self, mock_open, mock_json_load, mock_validator, mock_transform):
        """Test the validation process with mocks."""
        # Setup mock schema
        mock_schema = {"type": "object", "properties": {"age": {"type": "integer", "minimum": 0}}}
        mock_json_load.return_value = mock_schema
        
        # Mock file operations
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_open.return_value = mock_file
        
        # Setup mock validator
        mock_validator_instance = MagicMock()
        mock_validator.return_value = mock_validator_instance
        
        # Setup validation results
        errors_df = pd.DataFrame([{
            'Index': 0,
            'Validation Result': 'FAIL',
            'Schema path': '$.age',
            'Invalid key': 'age',
            'Validation error': 'Value must be greater than or equal to 0',
            'Validator': 'minimum',
            'Validator value': 0
        }])
        
        mock_validator_instance.validate_schema.return_value = (
            errors_df,
            pd.DataFrame([{'success_count': 0, 'fail_count': 1, 'total': 1}])
        )
        
        # Mock transform
        mock_transform.return_value = errors_df
        
        # Create test data
        test_df = pd.DataFrame([{"name": "Test", "age": -1}])
        
        # Create test instance
        reporter = ValidationReporter(schema_path="dummy_schema.json", dataframe=test_df)
        
        # Verify the transform was called
        mock_transform.assert_called_once()
        
        # Verify the output
        assert hasattr(reporter, 'output')
        assert not reporter.output.empty

# Test SyntheticDataCombiner
class TestSyntheticDataCombiner:
    def test_combine_dataframes(self, temp_dir):
        """Test combining multiple JSON files into a single DataFrame."""
        # Create test JSON files
        data1 = [{"id": 1, "name": "Test1"}]
        data2 = [{"id": 2, "value": 100}]
        
        os.makedirs(os.path.join(temp_dir, "test_data"), exist_ok=True)
        
        with open(os.path.join(temp_dir, "test_data", "file1.json"), 'w', encoding='utf-8') as f:
            json.dump(data1, f)
        with open(os.path.join(temp_dir, "test_data", "file2.json"), 'w', encoding='utf-8') as f:
            json.dump(data2, f)
        
        # Test the combiner
        combiner = SyntheticDataCombiner(
            os.path.join(temp_dir, "test_data")
        )
        combined = combiner.combined_df
        
        # Check the results
        assert not combined.empty
        assert 'name' in combined.columns
        assert 'value' in combined.columns
        assert len(combined) == 1  # Both have id=1 and id=2, but different columns


# Test SchemaResolver


class TestSchemaResolver:
    def test_resolve_references(self, temp_dir):
        """Test reference resolution in schemas."""
        # Create test schema with reference
        ref_schema = {
            "definitions": {
                "person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"}
                    },
                    "required": ["name"]
                }
            },
            "type": "object",
            "properties": {
                "person": {"$ref": "#/definitions/person"}
            }
        }
        
        # Save schema to file
        schema_path = os.path.join(temp_dir, "schema.json")
        with open(schema_path, 'w') as f:
            json.dump(ref_schema, f)
        
        # Create resolver and test
        resolver = SchemaResolver(
            bundle_json_path="dummy.json",  # Not used in this test
            unresolved_dir=os.path.join(temp_dir, "unresolved"),
            resolved_output_dir=os.path.join(temp_dir, "resolved"),
            definitions_fn="_definitions.json",
            terms_fn="_terms.json"
        )
        
        # Create a test reference file
        ref_content = {
    "definitions": {
        "person": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name"]
        }
    }
}
        ref_path = os.path.join(temp_dir, "ref.json")
        with open(ref_path, 'w', encoding='utf-8') as f:
            json.dump(ref_content, f)
        
        # Test with a schema that has a reference to the file we just created
        schema_with_ref = {"$ref": "ref.json#/definitions/person"}
        resolved = resolver.resolve_references(schema_with_ref, ref_path)
        
        # Check that the reference was resolved
        assert "$ref" not in str(resolved)
        assert "properties" in resolved
        assert "name" in resolved["properties"]

    def test_combine_resolved_schemas(self, temp_dir):
        """Test combining multiple resolved schemas into one."""
        # Create test schema files
        os.makedirs(os.path.join(temp_dir, "resolved"), exist_ok=True)
        
        schema1 = {"schema1": {"type": "object", "properties": {"name": {"type": "string"}}}}
        schema2 = {"schema2": {"type": "object", "properties": {"age": {"type": "integer"}}}}
        
        with open(os.path.join(temp_dir, "resolved", "schema1.json"), 'w', encoding='utf-8') as f:
            json.dump(schema1, f)
        with open(os.path.join(temp_dir, "resolved", "schema2.json"), 'w', encoding='utf-8') as f:
            json.dump(schema2, f)
        
        # Create resolver and combine schemas
        resolver = SchemaResolver(
            bundle_json_path="dummy.json",
            unresolved_dir=os.path.join(temp_dir, "unresolved"),
            resolved_output_dir=os.path.join(temp_dir, "resolved"),
            definitions_fn="_definitions.json",
            terms_fn="_terms.json"
        )
        
        output_file = os.path.join(temp_dir, "combined_schema.json")
        resolver.combine_resolved_schemas(
            resolved_dir=os.path.join(temp_dir, "resolved"),
            output_dir=temp_dir,
            output_filename="combined_schema.json"
        )
        
        # Check if combined schema was created and contains both schemas
        assert os.path.exists(output_file)
        with open(output_file, 'r', encoding='utf-8') as f:
            combined = json.load(f)
        assert "schema1" in combined
        assert "schema2" in combined

# Test QuickValidateSynth

