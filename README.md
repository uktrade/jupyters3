# Jupyter S3

Jupyter Notebook Contents Manager for AWS S3.


## Installation

```
pip install jupyters3
```


## Configuration

To configure Jupyter Notebook to use JupterS3, you can add the following to your notebook config file.

```python
from jupters3 import JupyterS3
c = get_config()
c.NotebookApp.contents_manager_class = JupyterS3
```

You _must_ also set the following settings on `c.JupyterS3` in your config file. None of them are optional.

| Setting | Description | Example |
| --- | --- | --- |
| `aws_s3_host`  | The hostname of the AWS S3 bucket. Typically, this is of the form `<aws_s3_bucket>.s3.<aws_region>.amazonaws.com`. | `'my-example-bucket.s3.eu-west-1.amazonaws.com'` |
| `aws_region` | The AWS region in which the bucket is located. This is used to sign the requests. | `'eu-west-1'` |
| `aws_access_key_id` | The ID of the AWS access key used to sign the requests to the AWS S3 API. | _ommitted_ |
| `aws_secret_access_key` | The secret part of the AWS access key used to sign the requests to the AWS S3 API. | _ommitted_ |
| `prefix` | The prefix to all keys used to store notebooks and checkpoints. This can be the empty string `''`. If non-empty, typically this would end in a forward slash `/`. | `'some-prefix/`' |


## Differences from S3Contents

- There are no extra dependencies over those already required for Jupyter Notebook. Specifically, there is no virtual filesystem library such as S3FS used, boto3 is not used, and Tornado is used as the HTTP client.

- Checkpoints are also saved to S3, under the key `<file_name>/.checkpoints/`.

- Multiple checkpoints are saved.

- The event loop is mostly not blocked during requests to S3. There are some exceptions due to Jupyter Notebook expecting certain requests to block.

- Uploading arbitrary files, such as JPEGs, and viewing them in Jupyter or downloading them, works.

- Copying and renaming files don't download or re-upload object data from or to S3. "PUT Object - Copy" is used instead.

- Requests to S3 are host-style, using a custom domain for the bucket, rather than path-style.

- All objects stored in S3 are encrypted with `AES256` encryption. This is not configurable: it is not possible to turn this off or to encrypt using a specific KMS key.

- AWS roles are not supported, although this may change.
