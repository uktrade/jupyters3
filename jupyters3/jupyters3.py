
import asyncio
import base64
from collections import namedtuple
import datetime
import hashlib
import hmac
import itertools
import json
import mimetypes
import os
import threading
import re
import time
import urllib
import xml.etree.ElementTree as ET

from tornado import gen
from tornado.httpclient import (
    AsyncHTTPClient,
    HTTPError as HTTPClientError,
    HTTPRequest,
)
from tornado.ioloop import IOLoop
from tornado.locks import Lock
from tornado.web import HTTPError as HTTPServerError
from traitlets.config.configurable import Configurable
from traitlets import (
    Dict,
    Unicode,
    Instance,
    TraitType,
    Type,
    default,
)

from nbformat.v4 import new_notebook
from notebook.services.contents.manager import (
    ContentsManager,
)

DIRECTORY_SUFFIX = '.s3keep'
NOTEBOOK_SUFFIX = '.ipynb'
CHECKPOINT_SUFFIX = '.checkpoints'
UNTITLED_NOTEBOOK = 'Untitled'
UNTITLED_FILE = 'Untitled'
UNTITLED_DIRECTORY  = 'Untitled Folder'
AWS_S3_PUT_HEADERS = {
    'x-amz-server-side-encryption': 'AES256',
}

Context = namedtuple('Context', [
    'logger', 'prefix', 'region', 's3_bucket', 's3_host', 's3_auth',
    'multipart_uploads',
])
AwsCreds = namedtuple('AwsCreds', [
    'access_key_id', 'secret_access_key', 'pre_auth_headers',
])


class ExpiringDict:

    def __init__(self, seconds):
        self._seconds = seconds
        self._store = {}

    def _remove_old_keys(self, now):
        self._store = {
            key: (expires, value)
            for key, (expires, value) in self._store.items()
            if expires > now
        }

    def __getitem__(self, key):
        now = int(time.time())
        self._remove_old_keys(now)
        return self._store[key][1]

    def __setitem__(self, key, value):
        now = int(time.time())
        self._remove_old_keys(now)
        self._store[key] = (now + self._seconds, value)

    def __delitem__(self, key):
        now = int(time.time())
        self._remove_old_keys(now)
        del self._store[key]


class Datetime(TraitType):
    klass = datetime.datetime
    default_value = datetime.datetime(1900, 1, 1)


class JupyterS3Authentication(Configurable):

    def get_credentials(self):
        raise NotImplementedError()


class JupyterS3SecretAccessKeyAuthentication(JupyterS3Authentication):

    aws_access_key_id = Unicode(config=True)
    aws_secret_access_key = Unicode(config=True)
    pre_auth_headers = Dict()

    @gen.coroutine
    def get_credentials(self):
        return AwsCreds(
            access_key_id=self.aws_access_key_id,
            secret_access_key=self.aws_secret_access_key,
            pre_auth_headers=self.pre_auth_headers,
        )


class JupyterS3ECSRoleAuthentication(JupyterS3Authentication):

    aws_access_key_id = Unicode()
    aws_secret_access_key = Unicode()
    pre_auth_headers = Dict()
    expiration = Datetime()

    @gen.coroutine
    def get_credentials(self):
        now = datetime.datetime.now()

        if now > self.expiration:
            request = HTTPRequest('http://169.254.170.2/' + os.environ['AWS_CONTAINER_CREDENTIALS_RELATIVE_URI'], method='GET')
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


class JupyterS3(ContentsManager):

    aws_s3_bucket = Unicode(config=True)
    aws_s3_host = Unicode(config=True)
    aws_region = Unicode(config=True)
    prefix = Unicode(config=True)

    authentication_class = Type(JupyterS3Authentication, config=True)
    authentication = Instance(JupyterS3Authentication)

    @default('authentication')
    def _default_authentication(self):
        return self.authentication_class(parent=self)

    # Do not use a checkpoints class: the rest of the system
    # only expects a ContentsManager
    checkpoints_class = None

    # Some of the write functions contain multiple S3 call
    # We do what we can to prevent bad things from happening
    write_lock = Instance(Lock)

    @default('write_lock')
    def _write_lock_default(self):
        return Lock()

    multipart_uploads = Instance(ExpiringDict)

    @default('multipart_uploads')
    def _multipart_uploads_default(self):
        return ExpiringDict(60 * 60 * 1)

    def is_hidden(self, path):
        return False

    # The next functions are not expected to be coroutines
    # or return futures. They have to block the event loop.

    def dir_exists(self, path):

        @gen.coroutine
        def dir_exists_async():
            return (yield _dir_exists(self._context(), path))

        return _run_sync_in_new_thread(dir_exists_async)

    def file_exists(self, path):

        @gen.coroutine
        def file_exists_async():
            return (yield _file_exists(self._context(), path))

        return _run_sync_in_new_thread(file_exists_async)

    def get(self, path, content, type, format=None):

        @gen.coroutine
        def get_async():
            return (yield _get(self._context(), path, content, type, format))

        return _run_sync_in_new_thread(get_async)

    @gen.coroutine
    def save(self, model, path):
        with (yield self.write_lock.acquire()):
            return (yield _save(self._context(), model, path))

    @gen.coroutine
    def delete(self, path):
        with (yield self.write_lock.acquire()):
            yield _delete(self._context(), path)

    @gen.coroutine
    def update(self, model, path):
        with (yield self.write_lock.acquire()):
            return (yield _rename(self._context(), path, model['path']))

    @gen.coroutine
    def new_untitled(self, path='', type='', ext=''):
        with (yield self.write_lock.acquire()):
            return (yield _new_untitled(self._context(), path, type, ext))

    @gen.coroutine
    def new(self, model, path):
        with (yield self.write_lock.acquire()):
            return (yield _new(self._context(), model, path))

    @gen.coroutine
    def copy(self, from_path, to_path):
        with (yield self.write_lock.acquire()):
            return (yield _copy(self._context(), from_path, to_path))

    @gen.coroutine
    def create_checkpoint(self, path):
        with (yield self.write_lock.acquire()):
            return (yield _create_checkpoint(self._context(), path))

    @gen.coroutine
    def restore_checkpoint(self, checkpoint_id, path):
        with (yield self.write_lock.acquire()):
            return (yield _restore_checkpoint(self._context(), checkpoint_id, path))

    @gen.coroutine
    def list_checkpoints(self, path):
        return (yield _list_checkpoints(self._context(), path))

    @gen.coroutine
    def delete_checkpoint(self, checkpoint_id, path):
        with (yield self.write_lock.acquire()):
            return (yield _delete_checkpoint(self._context(), checkpoint_id, path))

    def _context(self):
        return Context(
            logger=self.log,
            region=self.aws_region,
            s3_bucket=self.aws_s3_bucket,
            s3_host=self.aws_s3_host,
            s3_auth=self.authentication.get_credentials,
            prefix=self.prefix,
            multipart_uploads=self.multipart_uploads,
        )


# The documentation suggests that leading slashes in the
# path are not present, but they are, mostly
def _key(context, path):
    return context.prefix + path.lstrip('/')


def _path(context, key):
    return '/' + key[len(context.prefix):]


def _final_path_component(key_or_path):
    return key_or_path.split('/')[-1]


# The sort keys keep the UI as reasonable as possible with long running
# actions acting on multiple objects, including if things fail in the middle
def _copy_sort_key(key):
    return (key.count('/'), 0 if key.endswith('/' + DIRECTORY_SUFFIX) else 1)


def _delete_sort_key(key):
    return tuple(-1 * key_i for key_i in _copy_sort_key(key))


# We don't save type/format to S3, so we do some educated guesswork
# as to the types/formats of returned values.
@gen.coroutine
def _type_from_path(context, path):
    type = \
        'notebook' if path.endswith(NOTEBOOK_SUFFIX) else \
        'directory' if path == '/' or (yield _dir_exists(context, path)) else \
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


@gen.coroutine
def _dir_exists(context, path):
    return True if path == '/' else (yield _file_exists(context, path + '/' + DIRECTORY_SUFFIX))


@gen.coroutine
def _file_exists(context, path):

    @gen.coroutine
    def key_exists():
        key = _key(context, path)
        try:
            response = yield _make_s3_request(context, 'HEAD', '/' + key, {}, {}, b'')
        except HTTPClientError as exception:
            if exception.response.code != 404:
                raise HTTPServerError(exception.response.code, 'Error checking if S3 exists')
            response = exception.response

        return response.code == 200

    return False if path == '/' else (yield key_exists())


@gen.coroutine
def _exists(context, path):
    return (yield _file_exists(context, path)) or (yield _dir_exists(context, path)) 


@gen.coroutine
def _get(context, path, content, type, format):
    type_to_get = type if type is not None else (yield _type_from_path(context, path))
    format_to_get = format if format is not None else _format_from_type_and_path(context, type_to_get, path)
    return (yield GETTERS[(type_to_get, format_to_get)](context, path, content))


@gen.coroutine
def _get_notebook(context, path, content):
    return (yield _get_any(context, path, content, 'notebook', None, 'json', lambda file_bytes: json.loads(file_bytes.decode('utf-8'))))


@gen.coroutine
def _get_file_base64(context, path, content):
    return (yield _get_any(context, path, content, 'file', 'application/octet-stream', 'base64', lambda file_bytes: base64.b64encode(file_bytes).decode('utf-8')))


@gen.coroutine
def _get_file_text(context, path, content):
    return (yield _get_any(context, path, content, 'file', 'text/plain', 'text', lambda file_bytes: file_bytes.decode('utf-8')))


@gen.coroutine
def _get_any(context, path, content, type, mimetype, format, decode):
    method = 'GET' if content else 'HEAD'
    key = _key(context, path)
    response = yield _make_s3_request(context, method, '/' + key, {}, {}, b'')
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


@gen.coroutine
def _get_directory(context, path, content):
    key = _key(context, path)
    key_prefix = key if (key == '' or key[-1] == '/') else (key + '/')
    keys, directories = \
        (yield _list_immediate_child_keys_and_directories(context, key_prefix)) if content else \
        ([], [])

    all_keys = {key for (key, _) in keys}

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
            if directory not in all_keys
        ] + [
            {
                'type': _type_from_path_not_directory(key),
                'name': _final_path_component(key),
                'path': _path(context, key),
                'last_modified': last_modified,
            }
            for (key, last_modified) in keys
            if not key.endswith('/' + DIRECTORY_SUFFIX)
        ]) if content else None
    }


@gen.coroutine
def _save(context, model, path):
    type_to_save = model['type'] if 'type' in model else (yield _type_from_path(context, path))
    format_to_save = model['format'] if 'format' in model else _format_from_type_and_path(context, type_to_save, path)
    return (yield SAVERS[(type_to_save, format_to_save)](
        context,
        model['chunk'] if 'chunk' in model else None,
        model['content'] if 'content' in model else None,
        path,
    ))


@gen.coroutine
def _save_notebook(context, chunk, content, path):
    return (yield _save_any(context, chunk, json.dumps(content).encode('utf-8'), path, 'notebook', None))


@gen.coroutine
def _save_file_base64(context, chunk, content, path):
    return (yield _save_any(context, chunk, base64.b64decode(content.encode('utf-8')), path, 'file', 'application/octet-stream'))


@gen.coroutine
def _save_file_text(context, chunk, content, path):
    return (yield _save_any(context, chunk, content.encode('utf-8'), path, 'file', 'text/plain'))


@gen.coroutine
def _save_directory(context, chunk, content, path):
    return (yield _save_any(context, chunk, b'', path + '/' + DIRECTORY_SUFFIX, 'directory', None))


@gen.coroutine
def _save_any(context, chunk, content_bytes, path, type, mimetype):
    response = \
        (yield _save_bytes(context, content_bytes, path, type, mimetype)) if chunk is None else \
        (yield _save_chunk(context, chunk, content_bytes, path, type, mimetype))

    return response


@gen.coroutine
def _save_chunk(context, chunk, content_bytes, path, type, mimetype):
    # Chunks are 1-indexed
    if chunk == 1:
        context.multipart_uploads[path] = []
    context.multipart_uploads[path].append(content_bytes)

    # -1 is the last chunk
    if chunk == -1:
        combined_bytes = b''.join(context.multipart_uploads[path])
        del context.multipart_uploads[path]
        return (yield _save_bytes(context, combined_bytes, path, type, mimetype))
    else:
        return _saved_model(path, type, mimetype, datetime.datetime.now())


@gen.coroutine
def _save_bytes(context, content_bytes, path, type, mimetype):
    key = _key(context, path)
    response = yield _make_s3_request(context, 'PUT', '/' + key, {}, AWS_S3_PUT_HEADERS, content_bytes)

    last_modified_str = response.headers['Date']
    last_modified = datetime.datetime.strptime(last_modified_str, "%a, %d %b %Y %H:%M:%S GMT")
    return _saved_model(path, type, mimetype, last_modified)


def _saved_model(path, type, mimetype, last_modified):
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


@gen.coroutine
def _increment_filename(context, filename, path='', insert=''):
    basename, dot, ext = filename.partition('.')
    suffix = dot + ext

    for i in itertools.count():
        insert_i = f'{insert}{i}' if i else ''
        name = f'{basename}{insert_i}{suffix}'
        if not (yield _exists(context, f'/{path}/{name}')):
            break
    return name


def _checkpoint_path(path, checkpoint_id):
    return path + '/' + CHECKPOINT_SUFFIX + '/' + checkpoint_id


@gen.coroutine
def _create_checkpoint(context, path):
    model = yield _get(context, path, content=True, type=None, format=None)
    type = model['type']
    content = model['content']
    format = model['format']

    checkpoint_id = str(int(time.time() * 1000000))
    checkpoint_path = _checkpoint_path(path, checkpoint_id)
    yield SAVERS[(type, format)](context, None, content, checkpoint_path)
    # This is a new object, so shouldn't be any eventual consistency issues
    checkpoint = yield GETTERS[(type, format)](context, checkpoint_path, False)
    return {
        'id': checkpoint_id,
        'last_modified': checkpoint['last_modified'],
    }


@gen.coroutine
def _get_model_at_checkpoint(context, type, checkpoint_id, path):
    format = _format_from_type_and_path(context, type, path)
    checkpoint_path = _checkpoint_path(path, checkpoint_id)
    return (yield GETTERS[(type, format)](context, checkpoint_path, True))


@gen.coroutine
def _restore_checkpoint(context, checkpoint_id, path):
    type = (yield _get(context, path, content=False, type=None, format=None))['type']
    model = yield _get_model_at_checkpoint(context, type, checkpoint_id, path)
    yield _save(context, model, path)


@gen.coroutine
def _delete_checkpoint(context, checkpoint_id, path):
    checkpoint_path = _checkpoint_path(path, checkpoint_id)
    yield _delete(context, checkpoint_path)


@gen.coroutine
def _list_checkpoints(context, path):
    key_prefix = _key(context, path + '/' + CHECKPOINT_SUFFIX + '/')
    keys, _ = yield _list_immediate_child_keys_and_directories(context, key_prefix)
    return [
        {
            'id': key[(key.rfind('/' + CHECKPOINT_SUFFIX + '/') + len('/' + CHECKPOINT_SUFFIX + '/')):],
            'last_modified': last_modified,
        }
        for key, last_modified in keys
    ]


@gen.coroutine
def _rename(context, old_path, new_path):
    if (not (yield _exists(context, old_path))):
        raise HTTPServerError(400, "Source does not exist")

    if (yield _exists(context, new_path)):
        raise HTTPServerError(400, "Target already exists")

    type = yield _type_from_path(context, old_path)
    old_key = _key(context, old_path)
    new_key = _key(context, new_path)

    def replace_key_prefix(string):
        return new_key + string[len(old_key):]

    object_key = \
        [] if type == 'directory' else \
        [(old_key, new_key)]

    renames = object_key + [
        (key, replace_key_prefix(key))
        for (key, _) in (yield _list_all_descendant_keys(context, old_key + '/'))
    ]

    # We can't really do a transaction on S3, and not sure if we can trust that on any error
    # from DELETE, that the DELETE hasn't happened: even checking if the file is still there
    # isn't bulletproof due to eventual consistency. So we risk duplicate files over risking
    # deleted files
    for (old_key, new_key) in object_key + sorted(renames, key=lambda k: _copy_sort_key(k[0])):
        yield _copy_key(context, old_key, new_key)

    for (old_key, _) in sorted(renames, key=lambda k: _delete_sort_key(k[0])) + object_key:
        yield _delete_key(context, old_key)

    return (yield _get(context, new_path, content=False, type=None, format=None))


@gen.coroutine
def _copy_key(context, old_key, new_key):
    source_bucket = context.s3_bucket
    copy_headers = {
        'x-amz-copy-source': f'/{source_bucket}/{old_key}',
        **AWS_S3_PUT_HEADERS,
    }
    yield _make_s3_request(context, 'PUT', '/' + new_key, {}, copy_headers, b'')


@gen.coroutine
def _delete(context, path):
    if not path:
        raise HTTPServerError(400, "Can't delete root")

    type = _type_from_path(context, path)
    root_key = _key(context, path)

    object_key = \
        [] if type == 'directory' else \
        [root_key]

    descendant_keys = [
        key
        for (key, _) in (yield _list_all_descendant_keys(context, root_key + '/'))
    ]

    for key in sorted(descendant_keys, key=_delete_sort_key) + object_key:
        yield _delete_key(context, key)


@gen.coroutine
def _delete_key(context, key):
    yield _make_s3_request(context, 'DELETE', '/' + key, {}, {}, b'')


@gen.coroutine
def _new_untitled(context, path, type, ext):
    if not (yield _dir_exists(context, path)):
        raise HTTPServerError(404, 'No such directory: %s' % path)

    model_type = \
        type if type else \
        'notebook' if ext == '.ipynb' else \
        'file'

    untitled = \
        UNTITLED_DIRECTORY if model_type == 'directory' else \
        UNTITLED_NOTEBOOK if model_type == 'notebook' else \
        UNTITLED_FILE
    insert = \
        ' ' if model_type == 'directory' else \
        ''
    ext = \
        '.ipynb' if model_type == 'notebook' else \
        ext

    name = yield _increment_filename(context, untitled + ext, path, insert=insert)
    path = u'{0}/{1}'.format(path, name)

    model = {
        'type': model_type,
    }
    return (yield _new(context, model, path))


@gen.coroutine
def _new(context, model, path):
    if model is None:
        model = {}

    model.setdefault('type', 'notebook' if path.endswith('.ipynb') else 'file')

    if 'content' not in model and model['type'] == 'notebook':
        model['content'] = new_notebook()
        model['format'] = 'json'
    elif 'content' not in model and model['type'] == 'file':
        model['content'] = ''
        model['format'] = 'text'

    return (yield _save(context, model, path))


@gen.coroutine
def _copy(context, from_path, to_path):
    model = yield _get(context, from_path, content=False, type=None, format=None)
    if model['type'] == 'directory':
        raise HTTPServerError(400, "Can't copy directories")

    from_dir, from_name = \
        from_path.rsplit('/', 1) if '/' in from_path else \
        ('', from_path)

    to_path = \
        to_path if to_path is not None else \
        from_dir

    if (yield _dir_exists(context, to_path)):
        copy_pat = re.compile(r'\-Copy\d*\.')
        name = copy_pat.sub(u'.', from_name)
        to_name = yield _increment_filename(context, name, to_path, insert='-Copy')
        to_path = u'{0}/{1}'.format(to_path, to_name)

    from_key = _key(context, from_path)
    to_key = _key(context, to_path)

    yield _copy_key(context, from_key, to_key)
    return {
        **model,
        'name': to_name,
        'path': to_path,
    }


@gen.coroutine
def _list_immediate_child_keys_and_directories(context, key_prefix):
    return (yield _list_keys(context, key_prefix, '/'))


@gen.coroutine
def _list_all_descendant_keys(context, key_prefix):
    return (yield _list_keys(context, key_prefix, ''))[0]


@gen.coroutine
def _list_keys(context, key_prefix, delimeter):
    common_query = {
        'max-keys': '1000',
        'list-type': '2',
    }

    @gen.coroutine
    def _list_first_page():
        query = {
            **common_query,
            'delimiter': delimeter,
            'prefix': key_prefix,
        }
        response = yield _make_s3_request(context, 'GET', '/', query, {}, b'')
        return _parse_list_response(response)

    @gen.coroutine
    def _list_later_page(token):
        query = {
             **common_query,
            'continuation-token': token,
        }
        response = yield _make_s3_request(context, 'GET', '/', query, {}, b'')
        return _parse_list_response(response)

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
                keys.append((key, last_modified))
            if el.tag == f'{namespace}CommonPrefixes':
                # Prefixes end in '/', which we strip off
                directories.append(_first_child_text(el, f'{namespace}Prefix')[:-1])
            if el.tag == f'{namespace}NextContinuationToken':
                next_token = el.text

        return (next_token, keys, directories)

    token, keys, directories = yield _list_first_page()
    while token:
        token, keys_page, directories_page = yield _list_later_page(context.aws_endpoint, token)
        keys.extend(keys_page)
        directories.extend(directories_page)

    return keys, directories


@gen.coroutine
def _make_s3_request(context, method, path, query, api_pre_auth_headers, payload):
    service = 's3'
    credentials = yield context.s3_auth()
    pre_auth_headers = {
        **api_pre_auth_headers,
        **credentials.pre_auth_headers,
    }
    headers = _aws_headers(
        service, credentials.access_key_id, credentials.secret_access_key,
        context.region, context.s3_host, method, path, query, pre_auth_headers, payload,
    )
    querystring = urllib.parse.urlencode(query, safe='~', quote_via=urllib.parse.quote)
    encoded_path = urllib.parse.quote(path, safe='/~')
    url = f'https://{context.s3_host}{encoded_path}' + (('?' + querystring) if querystring else '')

    request = HTTPRequest(url, allow_nonstandard_methods=True, method=method, headers=headers, body=payload)

    try:
        response = (yield AsyncHTTPClient().fetch(request))
    except HTTPClientError as exception:
        if exception.response.code != 404:
            context.logger.warning(exception.response.body)
        raise

    return response


def _aws_headers(service, access_key_id, secret_access_key,
                 region, host, method, path, query, pre_auth_headers, payload):
    algorithm = 'AWS4-HMAC-SHA256'

    now = datetime.datetime.utcnow()
    amzdate = now.strftime('%Y%m%dT%H%M%SZ')
    datestamp = now.strftime('%Y%m%d')
    credential_scope = f'{datestamp}/{region}/{service}/aws4_request'
    headers_lower = {
        header_key.lower().strip(): header_value.strip()
        for header_key, header_value in pre_auth_headers.items()
    }
    required_headers = ['host', 'x-amz-content-sha256', 'x-amz-date']
    signed_header_keys = sorted([header_key
                                 for header_key in headers_lower.keys()] + required_headers)
    signed_headers = ';'.join(signed_header_keys)
    payload_hash = hashlib.sha256(payload).hexdigest()

    def signature():
        def canonical_request():
            header_values = {
                **headers_lower,
                'host': host,
                'x-amz-content-sha256': payload_hash,
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

        date_key = sign(('AWS4' + secret_access_key).encode('utf-8'), datestamp)
        region_key = sign(date_key, region)
        service_key = sign(region_key, service)
        request_key = sign(service_key, 'aws4_request')
        return sign(request_key, string_to_sign).hex()

    return {
        **pre_auth_headers,
        'x-amz-date': amzdate,
        'x-amz-content-sha256': payload_hash,
        'Authorization': (
            f'{algorithm} Credential={access_key_id}/{credential_scope}, ' +
            f'SignedHeaders={signed_headers}, Signature=' + signature()
        ),
    }


def _run_sync_in_new_thread(func):
    result = None
    exception = None
    def _func():
        nonlocal result
        nonlocal exception
        asyncio.set_event_loop(asyncio.new_event_loop())
        try:
            result = IOLoop.current().run_sync(func)
        except BaseException as _exception:
            exception = _exception

    thread = threading.Thread(target=_func)
    thread.start()
    thread.join()

    if exception is not None:
        raise exception
    else:
        return result


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
