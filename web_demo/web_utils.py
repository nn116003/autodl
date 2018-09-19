from torchvision import get_image_backend
from PIL import Image
import io
import base64
import os


REPO_DIRNAME = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../')
UPLOAD_FOLDER = '/tmp/demos_uploads'
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'])

ERR_DICT = {
    "img_url_err":{"code":404,
                   "message":"Cannot open image from URL."},
    "img_file_err":{"code":404,
                    "message":"Cannot open uploaded image."},
    "img_classify_err":{"code":600,
                        "message":"Somethin went wrong when classifying the image..."}
}

def pil_loader(path):
    return Image.open(path).convert('RGB')

def default_loader(path):
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS
    )

def embed_image_html(image):
    """Creates an image embedded in HTML base64 format."""
    string_buf = io.BytesIO()
    image.save(string_buf, format='png')
    data = base64.b64encode(string_buf.getvalue()).decode("utf-8")
    return 'data:image/png;base64,' + data
    

