import boto3
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
import logging


# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

class IndexdMetadata:
    """
    A class for constructing the file metadata for indexd upload.
    """

    def __init__(self, json_data_path: str, bucket_name: str, bucket_subdir: str, n_rows: int = None):
        """
        Initializes the IndexdMetadata object.

        Args:
            json_data_path (str): The path to the JSON metadata file. This file should be metadata for an entity of type 'file'.
            bucket_name (str): The name of the S3 bucket.
            bucket_subdir (str): The subdirectory path within the S3 bucket, listed before the file location as specified in the metadata.
        """
        self.json_data_path = json_data_path
        self.bucket_name = bucket_name
        self.bucket_subdir = bucket_subdir
        self.n_rows = n_rows
        self._check_json_data_is_file()
    
        

    def _check_json_data_is_file(self):
        """
        Checks if the json_data_path filename contains '_file'.

        Raises:
            Exception: If the filename does not contain '_file'.
        """
        filename = os.path.basename(self.json_data_path)
        if '_file' not in filename:
            raise Exception(
                f"{filename} invalid | The json_data_path must be metadata for a file and contain the extension '_file'."
            )

    def read_json_data(self, json_data_path, nrows=None) -> dict:
        if nrows is None:
            nrows = self.n_rows
        try:
            with open(json_data_path, 'r') as f:
                data = json.load(f)
                if nrows is not None and isinstance(data, list):
                    data = data[:nrows]
                logger.info(f"JSON data read from {json_data_path}")
            return data
        except FileNotFoundError:
            raise Exception(f"JSON data file not found: {json_data_path}")
        except json.JSONDecodeError:
            raise Exception(f"Invalid JSON format in {json_data_path}")

    def write_json_data(self, data: dict, json_data_path: str):
        try:
            # Create directory if it does not exist
            os.makedirs(os.path.dirname(json_data_path), exist_ok=True)

            with open(json_data_path, 'w') as f:
                json.dump(data, f, indent=4)
                logger.info(f"JSON data written to {json_data_path}")
        except Exception as e:
            logger.error(f"Error writing JSON data to {json_data_path}: {e}")

    def create_s3_uri(self, file_path: str) -> str:
        """
        Constructs an S3 URI using the provided file path.

        Args:
            file_path (str): The file path as specified in the JSON metadata.

        Returns:
            str: The constructed S3 URI.
        """
        strip_file_path = file_path.lstrip('./')
        s3_uri = f"s3://{self.bucket_name}/{self.bucket_subdir}/{strip_file_path}"
        return s3_uri

    def get_file_key_path(self, s3_uri: str) -> str:
        return '/'.join(s3_uri.split('/')[3:])

    def pull_s3_md5sum(self, s3_uri: str) -> str:
        """
        Pulls the MD5 checksum from an S3 URI.

        Args:
            s3_uri (str): The S3 URI of the file.

        Returns:
            str: The MD5 checksum of the file.

        Raises:
            Exception: If there is an error pulling the MD5 checksum or if the ETag is not found.
        """
        s3 = boto3.client('s3')
        try:
            response = s3.head_object(
                Bucket=self.bucket_name, Key=self.get_file_key_path(s3_uri)
            )
            return response['ETag'].strip('"')
        except boto3.exceptions.Boto3Error as e:
            raise Exception(f"Failed to pull MD5 checksum for {s3_uri}, error: {e}")
        except KeyError:
            raise Exception(f"ETag not found in the response for {s3_uri}")

    def validate_s3_uri(self, s3_uri: str) -> str:
        """
        Constructs an S3 URI from the given file path and validates it by checking for an MD5 checksum.

        Args:
            file_path (str): The path of the file to be used in constructing the S3 URI.

        Returns:
            str: A valid S3 URI if the MD5 checksum is successfully retrieved.

        Raises:
            Exception: If the S3 URI is invalid or the MD5 checksum cannot be retrieved.
        """
        # testing if s3_uri is valid and returns an md5
        md5 = self.pull_s3_md5sum(s3_uri)
        if md5:
            return s3_uri
        logging.error(f"Invalid S3 URI: {s3_uri}")
        raise Exception(f"Invalid S3 URI: {s3_uri}")

    def generate_did(self, md5: str, prefix: str = "PREFIX") -> str:
        """
        Creates a DID (Digital Identifier) which is suitable for indexd based on the provided MD5 sum.

        Args:
            md5 (str): The MD5 checksum of the file.
            prefix (str): The prefix to use for the DID. Defaults to "PREFIX".

        Returns:
            str: A DID string formatted as 'prefix/uuid', where uuid is derived from the MD5 checksum.

        Raises:
            ValueError: If the provided MD5 is invalid and cannot be converted to a UUID.
        """
        try:
            md5_uuid = uuid.UUID(md5)
        except ValueError as e:
            raise ValueError(f"Invalid MD5 provided: {md5}. Error: {e}")
        return f"{prefix}/{md5_uuid}"

    def pull_filesize(self, s3_uri: str) -> int:
        """
        Uses the s3_uri to pull the file size from S3.

        Args:
            s3_uri (str): The S3 URI of the file.

        Returns:
            int: The size of the file in bytes.

        Raises:
            Exception: If there is an error pulling the file size or if the ContentLength is not found.
        """
        s3 = boto3.client('s3')
        try:
            response = s3.head_object(
                Bucket=self.bucket_name, Key=self.get_file_key_path(s3_uri)
            )
            return response['ContentLength']
        except boto3.exceptions.Boto3Error as e:
            raise Exception(f"Failed to pull file size for {s3_uri}, error: {e}")
        except KeyError:
            raise Exception(f"ContentLength not found in the response for {s3_uri}")

    def get_authz(self, program: str, project: str) -> list:
        """
        Constructs the authorization resource string based on the given program and project.

        Args:
            program (str): The program name for authorization.
            project (str): The project name for authorization.

        Returns:
            list: A list containing the constructed authorization resource string.
        """
        auth_string = f"programs/{program}/projects/{project}"
        return [auth_string]
    
    def generate_baseid(self, filename:str) -> str:
        """
        Generates a base ID (GUID) from the provided filename.

        Args:
            filename (str): The name of the file for which to generate a GUID.

        Returns:
            str: A GUID string generated from the filename.
        """
        baseid = uuid.uuid5(uuid.NAMESPACE_DNS, filename)
        return str(baseid)

    def construct_metadata(self, s3_uri: str, program: str, project: str) -> dict:
        """
        Subprocess function that takes in an s3_uri, pulls S3 metadata into a structured format
        that is useful for indexd API upload. Program and project are used to define the authz resource.

        Args:
            s3_uri (str): The S3 URI of the file.
            program (str): The program name for authorization.
            project (str): The project name for authorization.

        Returns:
            dict: A dictionary containing structured metadata for indexd API upload.
        """
        self.validate_s3_uri(s3_uri)
        md5sum = self.pull_s3_md5sum(s3_uri)

        did = self.generate_did(md5sum)
        filesize = self.pull_filesize(s3_uri)
        authz = self.get_authz(program=program, project=project)
        file_key_path = self.get_file_key_path(s3_uri)
        file_name = file_key_path.split('/')[-1]

        indexd_metadata = {
            "hashes": {"md5": md5sum},
            "size": filesize,
            "did": did,
            "urls": [s3_uri],
            "file_name": file_name,
            "metadata": {},
            "baseid": self.generate_baseid(file_name),
            "acl": [],
            "urls_metadata": {s3_uri: {}},
            "version": None,
            "authz": authz,
            "content_created_date": None,
            "content_updated_date":None,
        }
        return indexd_metadata

    def update_metadata_indexd(self, json_data: dict, program: str, project: str):
        """
        Iterates through each object in the JSON data, constructs an S3 URI from the file path,
        validates the S3 URI, and generates a metadata dictionary using the S3 URI. This metadata
        dictionary includes a DID generated from the MD5 hash and contains necessary information
        for indexd upload. The method returns the updated JSON data and combined indexd metadata
        for uploading to indexd.

        Args:
            json_data (dict): The JSON data containing file information.
            program (str): The program name for authorization.
            project (str): The project name for authorization.

        Returns:
            tuple: A tuple containing the updated JSON data and a list of combined indexd metadata.
        """
        total_objects = len(json_data)

        def process_object(obj):
            s3_uri = self.create_s3_uri(obj["file_path"])
            self.validate_s3_uri(s3_uri)
            indexd_metadata_dict = self.construct_metadata(s3_uri, program, project)
            obj['object_id'] = indexd_metadata_dict['did']
            return obj, s3_uri, indexd_metadata_dict

        combined_indexd_metadata = []

        with ThreadPoolExecutor() as executor:
            future_to_obj = {
                executor.submit(process_object, obj): obj for obj in json_data
            }
            for idx, future in enumerate(as_completed(future_to_obj), start=1):
                obj, s3_uri, indexd_metadata_dict = future.result()
                combined_indexd_metadata.append(indexd_metadata_dict)
                percentage_complete = (idx / total_objects) * 100
                logger.info(
                    f"Updating metadata with Object_ID | Processed {idx} of {total_objects} "
                    f"({percentage_complete:.2f}%): {s3_uri}"
                )

        logger.info("All S3 URIs processed.")
        return json_data, combined_indexd_metadata

    def n_missing_object_id(self, json_data: dict) -> dict:
        """
        Checks for any JSON objects that don't have an 'object_id' added and returns a dictionary
        with two keys: 'missing_uris' and 'missing_indexes'. 'missing_uris' contains the list of
        's3_uri' values for objects missing an 'object_id', and 'missing_indexes' contains the
        corresponding indexes of these objects in the input JSON data.

        Args:
            json_data (dict): The JSON data to check for missing 'object_id' entries.

        Returns:
            dict: A dictionary with 'missing_uris' and 'missing_indexes' to identify objects
                  without an 'object_id'.
        """
        output_dict = {
            "missing_uris": [],
            "missing_indexes": [],
        }
        for idx, obj in enumerate(json_data):
            if isinstance(obj, dict) and 'object_id' not in obj:
                output_dict['missing_uris'].append(obj.get('file_path', 'Unknown'))
                output_dict['missing_indexes'].append(idx)

        if len(output_dict['missing_indexes']) == 0:
            logger.info("SUCCESS: All JSON objects have 'object_id' entries.")
        else:
            logger.warning("WARNING: Not all JSON objects have 'object_id' entries.")
        return output_dict

