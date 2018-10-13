import base64
from collections import namedtuple
import datetime
import hashlib
import hmac
import json
import mimetypes
import threading
import time
import urllib
import xml.etree.ElementTree as ET

from tornado.httpclient import (
    HTTPClient,
    HTTPError,
    HTTPRequest,
)
from traitlets import (
    Unicode,
    Type,
)

from notebook.services.contents.manager import (
    ContentsManager,
)
from notebook.services.contents.checkpoints import (
    GenericCheckpointsMixin,
    Checkpoints,
)

DIRECTORY_SUFFIX = '.s3keep'
NOTEBOOK_SUFFIX = '.ipynb'
CHECKPOINT_SUFFIX = '__CHECKPOINTS__'
Context = namedtuple('Context', ['logger', 'aws_endpoint', 'prefix'])


class JupyterS3Checkpoints(GenericCheckpointsMixin, Checkpoints):

    def create_file_checkpoint(self, content, format, path):
        return _create_checkpoint(self.parent._context(), 'file', content, format, path)

    def create_notebook_checkpoint(self, nb, path):
        return _create_checkpoint(self.parent._context(), 'notebook', nb, 'json', path)

    def get_file_checkpoint(self, checkpoint_id, path):
        return _get_checkpoint(self.parent._context(), 'file', checkpoint_id, path)

    def get_notebook_checkpoint(self, checkpoint_id, path):
        return _get_checkpoint(self.parent._context(), 'notebook', checkpoint_id, path)

    def delete_checkpoint(self, checkpoint_id, path):
        _delete_checkpoint(self.parent._context(), checkpoint_id, path)

    def list_checkpoints(self, path):
        return _list_checkpoints(self.parent._context(), path)

    def rename_checkpoint(self, checkpoint_id, old_path, new_path):
        _rename_checkpoint(self.parent._context(), checkpoint_id, old_path, new_path)


def _checkpoint_path(path, checkpoint_id):
    return path + CHECKPOINT_SUFFIX + '/' + checkpoint_id


def _create_checkpoint(context, type, content, format, path):
    checkpoint_id = str(int(time.time() * 1000000))
    checkpoint_path = _checkpoint_path(path, checkpoint_id)
    SAVERS[(type, format)](context, content, checkpoint_path)
    # This is a new object, so shouldn't be any eventual consistency issues
    checkpoint = GETTERS[(type, format)](context, checkpoint_path, False)
    return {
        'id': checkpoint_id,
        'last_modified': checkpoint['last_modified'],
    }


def _get_checkpoint(context, type, checkpoint_id, path):
    format = _format_from_type_and_path(context, type, path)
    checkpoint_path = _checkpoint_path(path, checkpoint_id)
    return GETTERS[(type, format)](context, checkpoint_path, True)


def _delete_checkpoint(context, checkpoint_id, path):
    checkpoint_path = _checkpoint_path(path, checkpoint_id)
    _delete(context, checkpoint_path)


def _list_checkpoints(context, path):
    key_prefix = _key(context, path + CHECKPOINT_SUFFIX + '/')
    keys, _ = _list_immediate_child_keys_and_directories(context, key_prefix)
    return [
        {
            'id': key[(key.rfind(CHECKPOINT_SUFFIX + '/') + len(CHECKPOINT_SUFFIX + '/')):],
            'last_modified': last_modified,
        }
        for key, last_modified in keys
    ]


def _rename_checkpoint(context, checkpoint_id, old_path, new_path):
    old_checkpoint_path = _checkpoint_path(old_path, checkpoint_id)
    new_checkpoint_path = _checkpoint_path(new_path, checkpoint_id)
    _rename(context, old_checkpoint_path, new_checkpoint_path)


class JupyterS3(ContentsManager):

    aws_region = Unicode(config=True)
    aws_host = Unicode(config=True)
    aws_access_key_id = Unicode(config=True)
    aws_secret_access_key = Unicode(config=True)
    prefix = Unicode(config=True)

    checkpoints_class = Type(JupyterS3Checkpoints, config=True)

    def dir_exists(self, path):
        return _dir_exists(self._context(), path)

    def is_hidden(self, path):
        return False

    def file_exists(self, path=''):
        return _file_exists(self._context(), path)

    def get(self, path, content=True, type=None, format=None):
        type_to_get = type if type is not None else _type_from_path(self._context(), path)
        format_to_get = format if format is not None else _format_from_type_and_path(self._context(), type_to_get, path)
        return GETTERS[(type_to_get, format_to_get)](self._context(), path, content)

    def save(self, model, path):
        type_to_save = model['type'] if 'type' in model else _type_from_path(self._context(), path)
        format_to_save = model['format'] if 'format' in model else _format_from_type_and_path(self._context(), type_to_save, path)
        return SAVERS[(type_to_save, format_to_save)](self._context(), model['content'] if 'content' in model else None, path)

    def delete_file(self, path):
        return _delete(self._context(), path)

    def rename_file(self, old_path, new_path):
        return _rename(self._context(), old_path, new_path)

    def _context(self):
        return Context(
            logger=self.log,
            aws_endpoint={
                'region': self.aws_region,
                'host': self.aws_host,
                'access_key_id': self.aws_access_key_id,
                'secret_access_key': self.aws_secret_access_key,
            },
            prefix=self.prefix,
        )


# The documentation suggests that leading slashes in the
# path are not present, but they sometimes seem to be
def _key(context, path):
    return context.prefix + path.lstrip('/')


def _path(context, key):
    return '/' + key[len(context.prefix):]


def _final_path_component(key_or_path):
    return key_or_path.split('/')[-1]


# We don't save type/format to S3, so we do some educated guesswork
# as to the types/formats of returned values.
def _type_from_path(context, path):
    type = \
        'notebook' if path.endswith(NOTEBOOK_SUFFIX) else \
        'directory' if path == '/' or _dir_exists(context, path) else \
        'file'
    return type


def _format_from_type_and_path(context, type, path):
    type = \
        'json' if type == 'notebook' else \
        'json' if type == 'directory' else \
        'text' if mimetypes.guess_type(path)[0] == 'text/plain' else \
        'base64'
    return type


def _type_from_path_not_directory(path):
    type = \
        'notebook' if path.endswith(NOTEBOOK_SUFFIX) else \
        'file'
    return type


def _dir_exists(context, path):
    return True if (path == '/' or path == '') else _file_exists(context, path + '/' + DIRECTORY_SUFFIX)


def _file_exists(context, path):
    def key_exists():
        key = _key(context, path)
        try:
            response = _make_s3_request(context, 'HEAD', '/' + key, {}, {}, b'')
        except HTTPError as exception:
            if exception.response.code != 404:
                raise
            response = exception.response

        return response.code == 200

    return False if path == '/' else key_exists()


def _get_notebook(context, path, content):
    return _get(context, path, content, 'notebook', None, 'json', lambda file_bytes: json.loads(file_bytes.decode('utf-8')))


def _get_file_base64(context, path, content):
    return _get(context, path, content, 'file', 'application/octet-stream', 'base64', lambda file_bytes: base64.b64encode(file_bytes).decode('utf-8'))


def _get_file_text(context, path, content):
    return _get(context, path, content, 'file', 'text/plain', 'text', lambda file_bytes: file_bytes.decode('utf-8'))


def _get(context, path, content, type, mimetype, format, decode):
    method = 'GET' if content else 'HEAD'
    key = _key(context, path)
    response = _make_s3_request(context, method, '/' + key, {}, {}, b'')
    file_bytes = response.body
    last_modified_str = response.headers['Last-Modified']
    last_modified = datetime.datetime.strptime(last_modified_str, "%a, %d %b %Y %H:%M:%S GMT")
    return {
        'name': _final_path_component(path),
        'path': path,
        'type': type,
        'mimetype': mimetype,
        'writable': True,
        'last_modified': last_modified, 
        'created': last_modified,
        'format': format if content else None,  
        'content': decode(file_bytes) if content else None,
    }  


def _get_directory(context, path, content):
    key = _key(context, path)
    key_prefix = key if (key == '' or key[-1] == '/') else (key + '/')
    keys, directories = \
        _list_immediate_child_keys_and_directories(context, key_prefix) if content else \
        ([], [])
    return {
        'name': _final_path_component(path),
        'path': path,
        'type': 'directory',
        'mimetype': None,
        'writable': True,
        'last_modified': datetime.datetime.fromtimestamp(86400), 
        'created': datetime.datetime.fromtimestamp(86400),
        'format': 'json' if content else None,
        'content': ([
            {
                'type': 'directory',
                'name': _final_path_component(directory),
                'path': _path(context, directory),
            }
            for directory in directories
            if CHECKPOINT_SUFFIX not in directory
        ] + [
            {
                'type': _type_from_path_not_directory(key),
                'name': _final_path_component(key),
                'path': _path(context, key),
                'last_modified': last_modified,
            }
            for (key, last_modified) in keys
        ]) if content else None
    }


def _save_notebook(context, content, path):
    return _save(context, json.dumps(content).encode('utf-8'), path, 'notebook', None)


def _save_file_base64(context, content, path):
    return _save(context, base64.b64decode(content.encode('utf-8')), path, 'file', 'application/octet-stream')


def _save_file_text(context, content, path):
    return _save(context, content.encode('utf-8'), path, 'file', 'text/plain')


def _save_directory(context, content, path):
    return _save(context, b'', path + '/' + DIRECTORY_SUFFIX, 'directory', None)


def _save(context, content_bytes, path, type, mimetype):
    key = _key(context, path)
    response = _make_s3_request(context, 'PUT', '/' + key, {}, {}, content_bytes)
    last_modified_str = response.headers['Date']
    last_modified = datetime.datetime.strptime(last_modified_str, "%a, %d %b %Y %H:%M:%S GMT")
    return {
        'name': _final_path_component(path),
        'path': path,
        'type': type,
        'mimetype': mimetype,
        'writable': True,
        'last_modified': last_modified, 
        'created': last_modified,
        'format': None,
        'content': None,
    }


def _rename(context, old_path, new_path):
    type = _type_from_path(context, old_path)
    old_key = _key(context, old_path)
    new_key = _key(context, new_path)

    def replace_key_prefix(string):
        return new_key + string[len(old_key):]

    renames = [
        (key, replace_key_prefix(key))
        for (key, _) in _list_all_descendant_keys(context, old_key + '/')
    ] if type == 'directory' else [
        (old_key, new_key)
    ]

    for (old_key, new_key) in renames:
        _rename_key(context, old_key, new_key)


def _rename_key(context, old_key, new_key):
    source_bucket = context.aws_endpoint['host'].split('.')[0]
    copy_headers = {
        'x-amz-copy-source': f'/{source_bucket}/{old_key}',
    }
    _make_s3_request(context, 'PUT', '/' + new_key, {}, copy_headers, b'')
    # We can't really do a transaction on S3, and not sure if we can trust that on any error
    # from DELETE, that the DELETE hasn't happened: even checking if the file is still there
    # isn't bulletproof due to eventual consistency. So we risk duplicate files over risking
    # deleted files
    _make_s3_request(context, 'DELETE', '/' + old_key, {}, {}, b'')


def _delete(context, path):
    type = _type_from_path(context, path)
    key = _key(context, path)

    keys = \
        _list_all_descendant_keys(context, '/' + key + '/') if type == 'directory' else \
        [(key, None)]

    for (key, _) in keys:
        _make_s3_request(context, 'DELETE', '/' + key, {}, {}, b'')


def _list_immediate_child_keys_and_directories(context, key_prefix):
    return _list_keys(context, key_prefix, '/', ['/' + DIRECTORY_SUFFIX])


def _list_all_descendant_keys(context, key_prefix):
    return _list_keys(context, key_prefix, '', [])[0]


def _list_keys(context, key_prefix, delimeter, omit):
    common_query = {
        'max-keys': '1000',
        'list-type': '2',
    }

    def _list_first_page():
        query = {
            **common_query,
            'delimiter': delimeter,
            'prefix': key_prefix,
        }
        return _parse_list_response(_make_s3_request(context, 'GET', '/', query, {}, b''))

    def _list_later_page(token):
        query = {
             **common_query,
            'continuation-token': token,
        }
        return _parse_list_response(_make_s3_request(context, 'GET', '/', query, {}, b''))

    def _first_child_text(el, tag):
        for child in el:
            if child.tag == tag:
                return child.text

    def _parse_list_response(response):
        namespace = '{http://s3.amazonaws.com/doc/2006-03-01/}'
        root = ET.fromstring(response.body)
        next_token = ''
        keys = []
        directories = []
        for el in root:
            if el.tag == f'{namespace}Contents':
                key = _first_child_text(el, f'{namespace}Key')
                last_modified_str = _first_child_text(el, f'{namespace}LastModified')
                last_modified = datetime.datetime.strptime(last_modified_str, "%Y-%m-%dT%H:%M:%S.%fZ")
                if not any([key.endswith(o) for o in omit]):
                    keys.append((key, last_modified))
            if el.tag == f'{namespace}CommonPrefixes':
                # Prefixes end in '/', which we strip off
                directories.append(_first_child_text(el, f'{namespace}Prefix')[:-1])
            if el.tag == f'{namespace}NextContinuationToken':
                next_token = el.text

        return (next_token, keys, directories)

    token, keys, directories = _list_first_page()
    while token:
        token, keys_page, directories_page = _list_later_page(context.aws_endpoint, token)
        keys.extend(keys_page)
        directories.extend(directories_page)

    return keys, directories


def _make_s3_request(context, method, path, query, non_auth_headers, payload):
    service = 's3'

    auth_headers = _aws_auth_headers(service, context.aws_endpoint, method, path, query, non_auth_headers, payload)
    headers = {
        **non_auth_headers,
        **auth_headers,
    }

    url = f'https://{context.aws_endpoint["host"]}{path}'
    querystring = '&'.join([
        urllib.parse.quote(key, safe='~') + '=' + urllib.parse.quote(value, safe='~')
        for key, value in query.items()
    ])
    encoded_path = urllib.parse.quote(path, safe='/~')
    url = f'https://{context.aws_endpoint["host"]}{encoded_path}' + (('?' + querystring) if querystring else '')

    # Because of...
    #
    # - The ContentsManager API expects blocking functions, not coroutines.
    #
    # - The API functions are called from inside couritines, and so from inside
    #   a running event loop
    #
    # - In Tornado 5, which is required by JupyterHub > 0.9, calling HTTPClient
    #   causes errors since it tries to start an event loop, and there can't
    #   be more than one per thread
    #
    # ... we create the client in a another thread, and block this thread until
    # the requests in that thread complete. Blocking the event loop appears to
    # be unavoidable form the design of the ContentsManager API. A related issue
    # is at https://github.com/jupyter/notebook/issues/3537 , where there is
    # mention of changing the API to allow coroutines.
    #
    # Most of the Notebook codebase appears to use coroutines, so we stick with
    # tornado for consistency, rather than using requests.
    response = None
    exception = None
    def request():
        nonlocal response
        nonlocal exception
        try:
            request = HTTPRequest(url, allow_nonstandard_methods=True, method=method, headers=headers, body=payload)
            response = HTTPClient().fetch(request)
        except BaseException as e:
            exception = e

    thread = threading.Thread(target=request)
    thread.start()
    thread.join()

    if exception is not None:
        raise exception
    else:
        return response

def _aws_auth_headers(service, aws_endpoint, method, path, query, headers, payload):
    algorithm = 'AWS4-HMAC-SHA256'

    now = datetime.datetime.utcnow()
    amzdate = now.strftime('%Y%m%dT%H%M%SZ')
    datestamp = now.strftime('%Y%m%d')
    credential_scope = f'{datestamp}/{aws_endpoint["region"]}/{service}/aws4_request'
    headers_lower = {
        header_key.lower().strip(): header_value.strip()
        for header_key, header_value in headers.items()
    }
    signed_header_keys = sorted([header_key
                                 for header_key in headers_lower.keys()] + ['host', 'x-amz-date'])
    signed_headers = ';'.join([header_key for header_key in signed_header_keys])
    payload_hash = hashlib.sha256(payload).hexdigest()

    def signature():
        def canonical_request():
            header_values = {
                **headers_lower,
                'host': aws_endpoint['host'],
                'x-amz-date': amzdate,
            }

            canonical_uri = urllib.parse.quote(path, safe='/~')
            query_keys = sorted(query.keys())
            canonical_querystring = '&'.join([
                urllib.parse.quote(key, safe='~') + '=' + urllib.parse.quote(query[key], safe='~')
                for key in query_keys
            ])
            canonical_headers = ''.join([
                header_key + ':' + header_values[header_key] + '\n'
                for header_key in signed_header_keys
            ])

            return f'{method}\n{canonical_uri}\n{canonical_querystring}\n' + \
                   f'{canonical_headers}\n{signed_headers}\n{payload_hash}'

        def sign(key, msg):
            return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()

        string_to_sign = \
            f'{algorithm}\n{amzdate}\n{credential_scope}\n' + \
            hashlib.sha256(canonical_request().encode('utf-8')).hexdigest()

        date_key = sign(('AWS4' + aws_endpoint['secret_access_key']).encode('utf-8'), datestamp)
        region_key = sign(date_key, aws_endpoint['region'])
        service_key = sign(region_key, service)
        request_key = sign(service_key, 'aws4_request')
        return sign(request_key, string_to_sign).hex()

    return {
        'x-amz-date': amzdate,
        'x-amz-content-sha256': payload_hash,
        'Authorization': (
            f'{algorithm} Credential={aws_endpoint["access_key_id"]}/{credential_scope}, ' +
            f'SignedHeaders={signed_headers}, Signature=' + signature()
        ),
    }


GETTERS = {
    ('notebook', 'json'): _get_notebook,
    ('file', 'text'): _get_file_text,
    ('file', 'base64'):  _get_file_base64,
    ('directory', 'json'): _get_directory,
}


SAVERS = {
    ('notebook', 'json'): _save_notebook,
    ('file', 'text'): _save_file_text,
    ('file', 'base64'):  _save_file_base64,
    ('directory', 'json'): _save_directory,
}
