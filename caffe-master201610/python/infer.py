import numpy as np
from PIL import Image

import caffe

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
im = Image.open('/media/iot/mydisk2/my/new/JPEGImages/111_1.jpg')
in_ = np.array(im, dtype=np.float32)
in_ = in_[:,:,::-1]
in_ -= np.array((104.00698793,116.66876762,122.67891434))
in_ = in_.transpose((2,0,1))

# load net
net = caffe.Net('/media/iot/mydisk2/wg_FCN_dir/fcn/voc-fcn8s/deploy.prototxt', '/media/iot/mydisk2/wg_FCN_dir/train_iter_5000.caffemodel', caffe.TEST)
# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_
# run net and take argmax for prediction
net.forward()
out = net.blobs['score'].data[0].argmax(axis=0)

labels = [np.array([i, j, k]) for i in range(0, 256, 80) for j in range(0, 256, 80) for k in range(0, 256, 80)]
cnt = 0
stdv = net.blobs["score"].data.max() * 7 / 10

for i in range(len(m)):
    for j in range(len(m[i])):
        if (m[i][j].max() > stdv):
            cnt += 1
            im[i][j] = labels[m[i][j].argmax()]

print cnt


img = Image.fromarray(im, "RGB")
img.save("show.png")

