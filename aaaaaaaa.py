import os

import numpy as np
import onnxruntime
import onnxruntime as rt
import tensorflow as tf
import tf2onnx
from sklearn.metrics import confusion_matrix
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.keras.applications import (densenet, inception_resnet_v2,
                                           inception_v3, mobilenet,
                                           mobilenet_v2, resnet, resnet_v2,
                                           vgg16)
from tensorflow.keras.applications.resnet50 import (ResNet50,
                                                    decode_predictions,
                                                    preprocess_input)
from tensorflow.keras.callbacks import (Callback, CSVLogger, EarlyStopping,
                                        ModelCheckpoint, TensorBoard)
#from tensorflow.keras.utils.generic_utils import CustomObjectScope
from tensorflow.keras.layers import (Activation, Conv2D, Dense,
                                     DepthwiseConv2D, Dropout, Flatten,
                                     GlobalAveragePooling2D, Input, Reshape)
from tensorflow.keras.metrics import top_k_categorical_accuracy
from tensorflow.keras.models import (Model, Sequential, load_model,
                                     model_from_json)
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import time
img_path = 'pic.jpg'

img = image.load_img(img_path, target_size=(224, 224))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

model = tf.keras.saving.load_model('mobilenet_7.h5')
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
output_path = model.name + ".onnx"

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
output_names = [n.name for n in model_proto.graph.output]
print("output_names", output_names)
providers = ['CPUExecutionProvider']

m = rt.InferenceSession(output_path, providers=providers)
t0 = time.time()
onnx_pred = m.run(output_names, {"input": x})
print("ONNX model took {:.2f} ms".format((time.time() - t0) * 1000))
idx_to_class={0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Happiness', 4: 'Neutral', 5: 'Sadness', 6: 'Surprise'}
print('ONNX Predicted:', idx_to_class[np.argmax(onnx_pred[0])])
print('ONNX Predicted:', onnx_pred[0])

