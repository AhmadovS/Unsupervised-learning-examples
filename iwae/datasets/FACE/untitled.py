from __future__ import division
from __future__ import print_function

from util import LoadData, Load, Save, DisplayPlot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
import scipy
import scipy.misc


inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('./toronto_face.npz')
def save_images(images, filename, **kwargs):
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)
    plot_images(images, ax, **kwargs)
    fig.patch.set_visible(False)
    ax.patch.set_visible(False)
    plt.savefig(filename)

def plot_images(images, ax, ims_per_row=5, padding=5, digit_dimensions=(28, 28),
                cmap=matplotlib.cm.binary, vmin=None, vmax=None):
    """Images should be a (N_images x pixels) matrix."""
    N_images = images.shape[0]
    N_rows = np.int32(np.ceil(float(N_images) / ims_per_row))
    pad_value = np.min(images.ravel())
    concat_images = np.full(((digit_dimensions[0] + padding) * N_rows + padding,
                             (digit_dimensions[1] + padding) * ims_per_row + padding), pad_value)
    for i in range(N_images):
        cur_image = np.reshape(images[i, :], digit_dimensions)
        row_ix = i // ims_per_row
        col_ix = i % ims_per_row
        row_start = padding + (padding + digit_dimensions[0]) * row_ix
        col_start = padding + (padding + digit_dimensions[1]) * col_ix
        concat_images[row_start: row_start + digit_dimensions[0],
                      col_start: col_start + digit_dimensions[1]] = cur_image
    cax = ax.matshow(concat_images, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    return cax

train = np.zeros((int(inputs_train.size / 2304), 784))
for i in range(int(inputs_train.size / 2304)):
	img = inputs_train[i]
	img = img.reshape((48, 48))	
	img = scipy.misc.imresize(img, (28, 28))
	img = img.reshape((784, ))
	train[i] = img

# test = np.zeros((int(inputs_test.size / 2304), 784))
# for i in range(int(inputs_test.size / 2304)):
#     img = inputs_test[i]
#     img = img.reshape((48, 48)) 
#     img = scipy.misc.imresize(img, (28, 28))
#     img = img.reshape((784, ))
#     test[i] = img

# np.save('toronto_face_train', train)
# np.save('toronto_face_test', test)

# img = inputs_train[42]
# img = img.reshape((48, 48))
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# plt.imshow(img, cmap='Greys')
# plt.savefig("damn.png")
# plt.close()