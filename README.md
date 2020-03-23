# Jupyter S3

Jupyter Notebook Contents Manager for AWS S3.


## Installation

```
pip install jupyters3
```


## Configuration

To configure Jupyter Notebook to use JupyterS3, you can add the following to your notebook config file.

```python
from jupyters3 import JupyterS3, JupyterS3SecretAccessKeyAuthentication
c = get_config()
c.NotebookApp.contents_manager_class = JupyterS3
```

and _must_ also set the following settings on `c.JupyterS3` in your config file.

| Setting | Description | Example |
| --- | --- | --- |
| `aws_region` | The AWS region in which the bucket is located | `'eu-west-1'` |
| `aws_s3_bucket` | The name of the S3 bucket. | `'my-example-bucket'` |
| `aws_s3_host`  | The hostname of the AWS S3 API. Typically, this is of the form `s3-<aws_region>.amazonaws.com`. | `'s3-eu-west-1.amazonaws.com'` |
| `prefix` | The prefix to all keys used to store notebooks and checkpoints. This can be the empty string `''`. If non-empty, typically this would end in a forward slash `/`. | `'some-prefix/`' |

You must also, either, authenticate using a secret key, in which case you must have the following configuration

```python
from jupyters3 import JupyterS3SecretAccessKeyAuthentication
c.JupyterS3.authentication_class = JupyterS3SecretAccessKeyAuthentication
```

_and_ the following settings on `c.JupyterS3SecretAccessKeyAuthentication`

| Setting | Description | Example |
| --- | --- | --- |
| `aws_access_key_id` | The ID of the AWS access key used to sign the requests to the AWS S3 API. | _ommitted_ |
| `aws_secret_access_key` | The secret part of the AWS access key used to sign the requests to the AWS S3 API. | _ommitted_ |

_or_ authenticate using a role in an ECS container, in which case you must have the following configuration

```python
from jupyters3 import JupyterS3ECSRoleAuthentication
c.JupyterS3.authentication_class = JupyterS3ECSRoleAuthentication
```

where JupyterS3ECSRoleAuthentication does not have configurable options, _or_ write your own authentication class, such as the one below for EC2/IAM role-based authentication

```python
import datetime
import json

from jupyters3 import (
    AwsCreds,
    JupyterS3Authentication,
)
from tornado import gen
from tornado.httpclient import (
    AsyncHTTPClient,
    HTTPError as HTTPClientError,
    HTTPRequest,
)

class JupyterS3EC2RoleAuthentication(JupyterS3Authentication):

    role_name = Unicode(config=True)
    aws_access_key_id = Unicode()
    aws_secret_access_key = Unicode()
    pre_auth_headers = Dict()
    expiration = Datetime()

    @gen.coroutine
    def get_credentials(self):
        now = datetime.datetime.now()

        if now > self.expiration:
            request = HTTPRequest('http://169.254.169.254/latest/meta-data/iam/security-credentials/' + self.role_name, method='GET')
            creds = json.loads((yield AsyncHTTPClient().fetch(request)).body.decode('utf-8'))
            self.aws_access_key_id = creds['AccessKeyId']
            self.aws_secret_access_key = creds['SecretAccessKey']
            self.pre_auth_headers = {
                'x-amz-security-token': creds['Token'],
            }
            self.expiration = datetime.datetime.strptime(creds['Expiration'], '%Y-%m-%dT%H:%M:%SZ')

        return AwsCreds(
            access_key_id=self.aws_access_key_id,
            secret_access_key=self.aws_secret_access_key,
            pre_auth_headers=self.pre_auth_headers,
        )
```

configured using

```python
c.JupyterS3.authentication_class = JupyterS3EC2RoleAuthentication
c.JupyterS3EC2RoleAuthentication.role_name = 'my-iam-role-name'
```


## Differences from S3Contents

- There are no extra dependencies over those already required for Jupyter Notebook. Specifically, there is no virtual filesystem library such as S3FS used, boto3 is not used, and Tornado is used as the HTTP client.

- Checkpoints are also saved to S3, under the key `<file_name>/.checkpoints/`.

- Multiple checkpoints are saved.

- The event loop is mostly not blocked during requests to S3. There are some exceptions due to Jupyter Notebook expecting certain requests to block.

- Uploading arbitrary files, such as JPEGs, and viewing them in Jupyter or downloading them, works.

- Copying and renaming files don't download or re-upload object data from or to S3. "PUT Object - Copy" is used instead.

- Folders are created using a 0 byte object with key suffix `/` (forward slash). A single forward slash suffix is consistent with both the AWS Console and AWS AppStream.
