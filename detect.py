from megengine import hub
model = hub.load(
    "megengine/models",
    "retinanet_res50_1x_800size",
    pretrained=True,
)
model.eval()

models_api = hub.import_module(
    "megengine/models",
    git_host="github.com",
)

# Download an example image from the megengine data website
import urllib
url, filename = ("https://data.megengine.org.cn/images/cat.jpg", "cat.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

# Read and pre-process the image
import cv2
image = cv2.imread("cat.jpg")

data, im_info = models_api.DetEvaluator.process_inputs(image, 800, 1333)
model.inputs["image"].set_value(data)
model.inputs["im_info"].set_value(im_info)

from megengine import jit
@jit.trace(symbolic=True)
def infer():
    predictions = model(model.inputs)
    return predictions

print(infer())
