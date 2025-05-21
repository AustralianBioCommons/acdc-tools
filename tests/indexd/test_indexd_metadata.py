from acdctools.indexd.indexd_metadata import IndexdMetadata
import pytest
from unittest.mock import mock_open, patch, MagicMock
import json
import boto3
import os

# Fixtures
@pytest.fixture
def metadata_file_content_list():
    content = [{
        "file_path": "./lipidomic/2025_01_17-lipidspecies/example_study-lipid-species-subject-990910020-sample-0001701.csv",
        "file_name": "example_study-lipid-species-subject-990910020-sample-0001701.csv",
        "file_size": 25279,
        "md5sum": "1cf21edb3f7a2c1e92e5c0b00bcb6c9f",
        "data_category": "summarised results",
        "cv": 0.2,
        "data_format": "csv",
        "imputed_data": "no",
        "individual_or_sum": "individual",
        "lipidomic_unit": "umol/mL",
        "data_type": "LC-MS",
        "type": "lipidomics_file",
        "lipidomics_assays": {
            "submitter_id": "lipidomics-assay-au-01-020-990910020"
        },
        "submitter_id": "lipidomics_file-example_study-1cf21edb3f7a2c1e92e5c0b00bcb6c9f"
    }]
    return content

@pytest.fixture
def init_IndexdMetadata():
    return IndexdMetadata("example/data/path/metadata_file.json", "example_bucket_name", "example_bucket_subdir")

def test_init_IndexdMetadata(init_IndexdMetadata):
    assert init_IndexdMetadata is not None


def test_read_json_data(init_IndexdMetadata, metadata_file_content_list):
    # Mock open to simulate reading the JSON file
    mock_json_str = json.dumps(metadata_file_content_list)
    with patch("builtins.open", mock_open(read_data=mock_json_str)):
        result = init_IndexdMetadata.read_json_data("example/data/path/metadata_file.json")
    assert result == metadata_file_content_list


@pytest.fixture
def indexd_submit_json():
    return [{
        "hashes": {
            "md5": "1cf21edb3f7a2c1e92e5c0b00bcb6c9f"
        },
        "size": 25288,
        "did": "PREFIX/1cf21edb-3f7a-2c1e-92e5-c0b00bcb6c9f",
        "urls": [
            "s3://example_bucket_name/example_bucket_subdir/lipidomic/2025_01_17-lipidspecies/example_study-lipid-species-subject-990910020-sample-0001701.csv"
        ],
        "file_name": "example_study-lipid-species-subject-990910020-sample-0001701.csv",
        "metadata": {},
        "baseid": "93af79c9-c04d-5eac-b5c4-37869d7fa8cd",
        "acl": [],
        "urls_metadata": {
            "s3://example_bucket_name/example_bucket_subdir/lipidomic/2025_01_17-lipidspecies/example_study-lipid-species-subject-990910020-sample-0001701.csv": {}
        },
        "version": None,
        "authz": [
            "programs/program1/projects/Example_Study"
        ],
        "content_created_date": None,
        "content_updated_date": None
    }]


def test_create_s3_uri(init_IndexdMetadata):
    # Test with a normal file path
    file_path = "lipidomic/2025_01_17-lipidspecies/example_study-lipid-species-subject-990910020-sample-0001701.csv"
    expected_uri = (
        "s3://example_bucket_name/example_bucket_subdir/"
        "lipidomic/2025_01_17-lipidspecies/example_study-lipid-species-subject-990910020-sample-0001701.csv"
    )
    result = init_IndexdMetadata.create_s3_uri(file_path)
    assert result == expected_uri

    # Test with a file path that starts with './'
    file_path_dot = "./lipidomic/2025_01_17-lipidspecies/example_study-lipid-species-subject-990910020-sample-0001701.csv"
    result_dot = init_IndexdMetadata.create_s3_uri(file_path_dot)
    assert result_dot == expected_uri

    # Test with a file path that starts with multiple './'
    file_path_multi_dot = "././lipidomic/2025_01_17-lipidspecies/example_study-lipid-species-subject-990910020-sample-0001701.csv"
    result_multi_dot = init_IndexdMetadata.create_s3_uri(file_path_multi_dot)
    assert result_multi_dot == expected_uri
    
def test_get_file_key_path(init_IndexdMetadata):
    # S3 URI with bucket and subdir
    s3_uri = "s3://example_bucket_name/example_bucket_subdir/lipidomic/2025_01_17-lipidspecies/file.csv"
    expected_key = "example_bucket_subdir/lipidomic/2025_01_17-lipidspecies/file.csv"
    result = init_IndexdMetadata.get_file_key_path(s3_uri)
    assert result == expected_key

    # S3 URI with only bucket (no subdir)
    s3_uri_no_subdir = "s3://example_bucket_name/file.csv"
    expected_key_no_subdir = "file.csv"
    result_no_subdir = init_IndexdMetadata.get_file_key_path(s3_uri_no_subdir)
    assert result_no_subdir == expected_key_no_subdir

    # S3 URI with nested subdirs
    s3_uri_nested = "s3://bucket/sub1/sub2/sub3/file.txt"
    expected_key_nested = "sub1/sub2/sub3/file.txt"
    result_nested = init_IndexdMetadata.get_file_key_path(s3_uri_nested)
    assert result_nested == expected_key_nested

import pytest

def test_generate_did_valid(init_IndexdMetadata):
    # A valid MD5 string (32 hex chars) can be used to create a UUID
    # We'll use a valid UUID string as MD5 for this test
    valid_md5 = "1cf21edb3f7a2c1e92e5c0b00bcb6c9f"
    # UUID accepts hex string without dashes
    expected_uuid = "1cf21edb-3f7a-2c1e-92e5-c0b00bcb6c9f"
    expected_prefix = "PREFIX"
    expected_did = f"{expected_prefix}/{expected_uuid}"
    result = init_IndexdMetadata.generate_did(valid_md5)
    assert result == expected_did

    # Test with a custom prefix
    custom_prefix = "MYPRJ"
    result_custom = init_IndexdMetadata.generate_did(valid_md5, prefix=custom_prefix)
    assert result_custom == f"{custom_prefix}/{expected_uuid}"

def test_generate_did_invalid(init_IndexdMetadata):
    # An invalid MD5 string (not a valid UUID)
    invalid_md5 = "notavalidmd5sum1234567890abcdef"
    with pytest.raises(ValueError) as excinfo:
        init_IndexdMetadata.generate_did(invalid_md5)
    assert "Invalid MD5 provided" in str(excinfo.value)


import pytest
from unittest.mock import patch, MagicMock

def test_pull_filesize_success(init_IndexdMetadata):
    s3_uri = "s3://example_bucket_name/example_bucket_subdir/lipidomic/2025_01_17-lipidspecies/file.csv"
    expected_filesize = 123456

    mock_s3 = MagicMock()
    mock_s3.head_object.return_value = {
        "ContentLength": expected_filesize
    }

    with patch("boto3.client", return_value=mock_s3):
        filesize = init_IndexdMetadata.pull_filesize(s3_uri)
        assert filesize == expected_filesize
        mock_s3.head_object.assert_called_once_with(
            Bucket=init_IndexdMetadata.bucket_name,
            Key=init_IndexdMetadata.get_file_key_path(s3_uri)
        )

def test_pull_filesize_boto3_error(init_IndexdMetadata):
    s3_uri = "s3://example_bucket_name/example_bucket_subdir/lipidomic/2025_01_17-lipidspecies/file.csv"

    mock_s3 = MagicMock()
    # Simulate boto3.exceptions.Boto3Error
    class DummyBoto3Error(Exception):
        pass
    mock_s3.head_object.side_effect = DummyBoto3Error("Boto3 error")

    with patch("boto3.client", return_value=mock_s3), \
         patch("boto3.exceptions.Boto3Error", DummyBoto3Error):
        with pytest.raises(Exception) as excinfo:
            init_IndexdMetadata.pull_filesize(s3_uri)
        assert "Failed to pull file size" in str(excinfo.value)

def test_pull_filesize_keyerror(init_IndexdMetadata):
    s3_uri = "s3://example_bucket_name/example_bucket_subdir/lipidomic/2025_01_17-lipidspecies/file.csv"

    mock_s3 = MagicMock()
    # Return dict without 'ContentLength'
    mock_s3.head_object.return_value = {}

    with patch("boto3.client", return_value=mock_s3):
        with pytest.raises(Exception) as excinfo:
            init_IndexdMetadata.pull_filesize(s3_uri)
        assert "ContentLength not found" in str(excinfo.value)

def test_get_authz_returns_correct_authz(init_IndexdMetadata):
    program = "my_program"
    project = "my_project"
    expected_authz = [f"programs/{program}/projects/{project}"]
    result = init_IndexdMetadata.get_authz(program, project)
    assert result == expected_authz

def test_generate_baseid_returns_uuid_string(init_IndexdMetadata):
    # The baseid should be a valid UUID string generated from the filename
    filename = "example_file.txt"
    baseid = init_IndexdMetadata.generate_baseid(filename)
    # Should be a string
    assert isinstance(baseid, str)
    # Should be a valid UUID
    import uuid
    try:
        uuid_obj = uuid.UUID(baseid)
    except ValueError:
        pytest.fail("Returned baseid is not a valid UUID string")
    # Should be deterministic for the same filename
    baseid2 = init_IndexdMetadata.generate_baseid(filename)
    assert baseid == baseid2
    # Should be different for a different filename
    baseid3 = init_IndexdMetadata.generate_baseid("another_file.txt")
    assert baseid != baseid3


import pytest
from unittest.mock import patch, MagicMock

@pytest.fixture
def s3_uri():
    return "s3://example_bucket_name/example_bucket_subdir/lipidomic/2025_01_17-lipidspecies/file.csv"

@pytest.fixture
def program():
    return "test_program"

@pytest.fixture
def project():
    return "test_project"

def test_construct_metadata_returns_expected_dict(init_IndexdMetadata, s3_uri, program, project):
    # Prepare mocks for all methods called inside construct_metadata
    md5sum = "dummy_md5"
    did = "dummy_did"
    filesize = 12345
    authz = ["programs/test_program/projects/test_project"]
    file_key_path = "example_bucket_subdir/lipidomic/2025_01_17-lipidspecies/file.csv"
    file_name = "file.csv"
    baseid = "dummy_baseid"

    with patch.object(init_IndexdMetadata, "validate_s3_uri") as mock_validate, \
         patch.object(init_IndexdMetadata, "pull_s3_md5sum", return_value=md5sum) as mock_md5, \
         patch.object(init_IndexdMetadata, "generate_did", return_value=did) as mock_did, \
         patch.object(init_IndexdMetadata, "pull_filesize", return_value=filesize) as mock_filesize, \
         patch.object(init_IndexdMetadata, "get_authz", return_value=authz) as mock_authz, \
         patch.object(init_IndexdMetadata, "get_file_key_path", return_value=file_key_path) as mock_key_path, \
         patch.object(init_IndexdMetadata, "generate_baseid", return_value=baseid) as mock_baseid:

        result = init_IndexdMetadata.construct_metadata(s3_uri, program, project)

    expected = {
        "hashes": {"md5": md5sum},
        "size": filesize,
        "did": did,
        "urls": [s3_uri],
        "file_name": file_name,
        "metadata": {},
        "baseid": baseid,
        "acl": [],
        "urls_metadata": {s3_uri: {}},
        "version": None,
        "authz": authz,
        "content_created_date": None,
        "content_updated_date": None,
    }
    assert result == expected
    mock_validate.assert_called_once_with(s3_uri)
    mock_md5.assert_called_once_with(s3_uri)
    mock_did.assert_called_once_with(md5sum)
    mock_filesize.assert_called_once_with(s3_uri)
    mock_authz.assert_called_once_with(program=program, project=project)
    mock_key_path.assert_called_once_with(s3_uri)
    mock_baseid.assert_called_once_with(file_name)



def test_update_metadata_indexd(init_IndexdMetadata, metadata_file_content_list, indexd_submit_json):
    # Mock S3 head_object response
    mock_s3 = MagicMock()
    mock_s3.head_object.return_value = {
        "ETag": "1cf21edb3f7a2c1e92e5c0b00bcb6c9f",
        "ContentLength": 25288
    }
    with patch("boto3.client", return_value=mock_s3):
        file_metadata_json, indexd_submit_json_result = init_IndexdMetadata.update_metadata_indexd(
            metadata_file_content_list, "program1", "Example_Study"
        )
    assert indexd_submit_json_result == indexd_submit_json


def test_write_json_data(init_IndexdMetadata):
    # Prepare test data and file path
    json_data = {"foo": "bar", "baz": [1, 2, 3]}
    dir_path = "tmp_path"
    basename = "test_indexd_metadata.json"
    file_path = os.path.join(dir_path, "indexd", basename)

    # Call the method
    init_IndexdMetadata.write_json_data(json_data, file_path)

    # Assert file was created and contents are correct
    assert os.path.exists(file_path)
    with open(file_path, "r") as f:
        loaded = json.load(f)
    assert loaded == json_data
