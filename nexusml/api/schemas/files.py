from marshmallow import fields
from marshmallow import validate
from marshmallow import validates
from marshmallow import validates_schema
from marshmallow import ValidationError
from marshmallow_enum import EnumField

from nexusml.api.schemas.base import BaseSchema
from nexusml.api.schemas.base import ImmutableResourceResponseSchema
from nexusml.api.schemas.base import PageSchema
from nexusml.api.schemas.base import ResourceRequestSchema
from nexusml.api.utils import config
from nexusml.enums import FileType
from nexusml.enums import TaskFileUse


class FileSchema(BaseSchema):
    filename = fields.String(required=True,
                             description='Name of the file. Prefixes can be used to represent directories. '
                             'WARNING: filenames might not be unique')
    size = fields.Integer(required=True, validate=validate.Range(min=1), description='File size in bytes')


class FileRequest(FileSchema, ResourceRequestSchema):
    pass


class FileResponse(FileSchema, ImmutableResourceResponseSchema):

    class PresignedPostSchema(BaseSchema):

        class Meta:
            include = {'fields': fields.Dict(description='Form fields and values to submit with the POST')}

        url = fields.String(required=True, description='URL to post to')

    download_url = fields.String(description='URL from which file content can be downloaded')
    upload_url = fields.Nested(PresignedPostSchema,
                               description='URL to which file content can be uploaded.'
                               '\nNot provided for large files. '
                               'Large files must be uploaded using a multipart upload.')


class OrganizationFileSchema(FileSchema):
    pass


class OrganizationFileRequest(OrganizationFileSchema, FileRequest):

    @validates('size')
    def validate_picture_size(self, size):
        # Note: we are assuming there is only one file use ("picture")
        max_pic_size = config.get('limits')['organizations']['picture_size']
        if max_pic_size > 1024:
            max_pic_size_str = str(round(max_pic_size / 1024)) + ' KB'
        elif max_pic_size > 1024**2:
            max_pic_size_str = str(round(max_pic_size / (1024**2))) + ' MB'
        else:
            max_pic_size_str = str(max_pic_size) + ' bytes'
        if size > max_pic_size:
            raise ValidationError(f'Maximum image size ({max_pic_size_str}) exceed')


class OrganizationFileResponse(OrganizationFileSchema, FileResponse):
    pass


class TaskFileSchema(FileSchema):

    _file_uses = ' | '.join([f'"{x.name.lower()}"' for x in TaskFileUse])

    class Meta:
        _file_types = ' | '.join([f'"{x.name.lower()}"' for x in FileType])
        include = {
            'type': EnumField(FileType, allow_none=True, description=f'File type: {_file_types}'),
        }

    use_for = EnumField(TaskFileUse, required=True, description=f'File use: {_file_uses}')


class TaskFileRequest(TaskFileSchema, FileRequest):

    @validates_schema
    def validate_use_and_type(self, data, **kwargs):
        # AI models
        if data['use_for'] == TaskFileUse.AI_MODEL and data['type'] is not None:
            raise ValidationError('Invalid AI model file')
        # Pictures
        if data['use_for'] == TaskFileUse.PICTURE and data['type'] not in [FileType.IMAGE, None]:
            raise ValidationError('Invalid picture')

    @validates_schema
    def validate_size(self, data, **kwargs):
        # Pictures
        max_pic_size = config.get('limits')['tasks']['picture_size']
        if data['use_for'] == TaskFileUse.PICTURE and data['size'] > max_pic_size:
            raise ValidationError(f'Maximum task icon size ({round(max_pic_size / (1024 ** 2))}) exceed')


class TaskFileResponse(TaskFileSchema, FileResponse):
    pass


class FilesPage(PageSchema):
    _files_page_description = 'Files in the requested page (or in the first page, if not specified)'
    data = fields.List(fields.Nested(FileResponse), required=True, description=_files_page_description)


class OrganizationFilesPage(FilesPage):
    data = fields.List(fields.Nested(OrganizationFileResponse),
                       required=True,
                       description=FilesPage._files_page_description)


class TaskFilesPage(FilesPage):
    data = fields.List(fields.Nested(TaskFileResponse), required=True, description=FilesPage._files_page_description)


class FilePartUploadRequest(BaseSchema):
    part_number = fields.Integer(required=True, validate=validate.Range(min=1), description='Part number')


class FilePartUploadResponse(BaseSchema):
    upload_url = fields.String(required=True, description='URL to which the specified file part can be uploaded')


class FileUploadCompletionRequest(BaseSchema):

    class UploadedPart(BaseSchema):
        part_number = fields.Integer(required=True, validate=validate.Range(min=1), description='Part number')
        etag = fields.String(required=True, validate=validate.Length(min=1), description='Entity tag (ETag)')

    uploaded_parts = fields.List(fields.Nested(UploadedPart),
                                 required=True,
                                 validate=validate.Length(min=1),
                                 description="Uploaded parts' entity tag (ETag)")
