import random
import numpy as np
import scipy
from scipy import ndimage
from PIL import Image, ImageEnhance, ImageOps
from tensorflow.python.keras.backend import image_data_format

# utility function for global contrast normalization
def normalize(train, test):
	# truncate rgb to 0-1
	train = train / 255.0
	test = test / 255.0

	# zero centering and std normalization
	mean = np.mean(train, axis=0)
	std = np.std(train, axis=0)
	epsilon = 1e-7 # avoid dividing by zero
	
	train = (train - mean) / (std + epsilon)
	test = (test - mean) / (std + epsilon)
	
	return train, test

# zero component analysis whitening (normalize data first)
def zca_whitening(train, test):
	cov = np.cov(train, rowvar=False) # covariance matrix
	u, s, _ = np.linalg.svd(cov) # singular value decomposition
	epsilon = 0.1 # whitening coefficient
	
	# build zca matrix
	zca_matrix = u.dot(np.diag(1.0 / np.sqrt(s + epsilon))).dot(u.T)
	train = zca_matrix.dot(train.T).T
	test = zca_matrix.dot(test.T).T
	
	return train, test

# per pixel principal component analysis whitening (normalize data first)
# needs more testing
def per_pixel_pca(train, test):
	cov = np.cov(train, rowvar=False) # covariance matrix
	eval, evec = np.linalg.eigh(cov) # eigenvalues and -vectors
	idx = np.argsort(eval)[::-1] # reverse indices
	
	# sort and get the best 3 eigenvectors
	eval = eval[idx]
	evec = evec[:, :3]
	features = (evec).column_stack().matrix()
	
	# apply PCA
	train = evec.T.dot(train.T).T
	test = evec.T.dot(test.T).T
	
	# normal distribution parameters
	mu = 0
	sigma = 0.1
	
	# scale eigenvalues w/ randomized normal distribution
	se = np.zeros((3, 1))
	se[0][0] = np.random.normal(mu, sigma) * eval[0]
	se[1][0] = np.random.normal(mu, sigma) * eval[1]
	se[2][0] = np.random.normal(mu, sigma) * eval[2]
	val = features * se.matrix()
	
	# add magnitudes to principal component multiples
	train[...,0] += val[0]
	train[...,1] += val[1]
	train[...,2] += val[2]
	
	test[...,0] += val[0]
	test[...,1] += val[1]
	test[...,2] += val[2]
	
	return train, test

# cutout & random erasing preprocessing function for ImageDataGenerator
# implementation by Yusuke Uchida & Kosuke Takeuchi https://github.com/yu4u/cutout-random-erasing
def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        if input_img.ndim == 3:
            img_h, img_w, img_c = input_img.shape
        elif input_img.ndim == 2:
            img_h, img_w = input_img.shape

        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            if input_img.ndim == 3:
                c = np.random.uniform(v_l, v_h, (h, w, img_c))
            if input_img.ndim == 2:
                c = np.random.uniform(v_l, v_h, (h, w))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w] = c

        return input_img

    return eraser

# cutmix augmentation ImageDataGenerator
# https://github.com/wakamezake/CutMix
class CutMixGenerator:
    def __init__(self, X_train, y_train, batch_size=32, alpha=0.2,
                 shuffle=True, data_gen=None):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(X_train)
        self.data_gen = data_gen

    def __call__(self):
        while True:
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))

            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                X, y = self.__data_generation(batch_ids)

                yield X, y

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        _, h, w, c = self.X_train.shape
        _, class_num = self.y_train.shape
        batch_index = batch_ids[:self.batch_size]
        rand_index = np.random.permutation(batch_index)
        X1 = self.X_train[batch_index]
        X2 = self.X_train[rand_index]
        y1 = self.y_train[batch_index]
        y2 = self.y_train[rand_index]
        lam = np.random.beta(self.alpha, self.alpha)

        bx1, by1, bx2, by2 = get_rand_bbox(w, h, lam)
        X1[:, bx1:bx2, by1:by2, :] = X2[:, bx1:bx2, by1:by2, :]
        X = X1
        y = y1 * lam + y2 * (1 - lam)

        if self.data_gen:
            for i in range(self.batch_size):
                X[i] = self.data_gen.random_transform(X[i])

        return X, y


def is_channel_last(image):
    channel = image.shape[2]
    assert len(image.shape) == 3
    assert channel == 3 or channel == 1
    assert image_data_format() == "channels_last"

def get_rand_bbox(width, height, l):
    r_x = np.random.randint(width)
    r_y = np.random.randint(height)
    r_l = np.sqrt(1 - l)
    r_w = np.int(width * r_l)
    r_h = np.int(height * r_l)
    bb_x_1 = np.clip(r_x - r_w // 2, 0, width)
    bb_y_1 = np.clip(r_y - r_h // 2, 0, height)
    bb_x_2 = np.clip(r_x + r_w // 2, 0, width)
    bb_y_2 = np.clip(r_y + r_h // 2, 0, height)
    return bb_x_1, bb_y_1, bb_x_2, bb_y_2


# autoaugment ImageDataGenerator
# https://github.com/4uiiurz1/keras-auto-augment
class AutoAugmentGenerator:
    def __init__(self, args):
        self.datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, fill_mode='constant', cval=0, horizontal_flip=True)

        self.means = np.array([0.4914009 , 0.48215896, 0.4465308])
        self.stds = np.array([0.24703279, 0.24348423, 0.26158753])

        self.args = args
        if args.auto_augment:
            self.policies = [
                ['Invert', 0.1, 7, 'Contrast', 0.2, 6],
                ['Rotate', 0.7, 2, 'TranslateX', 0.3, 9],
                ['Sharpness', 0.8, 1, 'Sharpness', 0.9, 3],
                ['ShearY', 0.5, 8, 'TranslateY', 0.7, 9],
                ['AutoContrast', 0.5, 8, 'Equalize', 0.9, 2],
                ['ShearY', 0.2, 7, 'Posterize', 0.3, 7],
                ['Color', 0.4, 3, 'Brightness', 0.6, 7],
                ['Sharpness', 0.3, 9, 'Brightness', 0.7, 9],
                ['Equalize', 0.6, 5, 'Equalize', 0.5, 1],
                ['Contrast', 0.6, 7, 'Sharpness', 0.6, 5],
                ['Color', 0.7, 7, 'TranslateX', 0.5, 8],
                ['Equalize', 0.3, 7, 'AutoContrast', 0.4, 8],
                ['TranslateY', 0.4, 3, 'Sharpness', 0.2, 6],
                ['Brightness', 0.9, 6, 'Color', 0.2, 8],
                ['Solarize', 0.5, 2, 'Invert', 0, 0.3],
                ['Equalize', 0.2, 0, 'AutoContrast', 0.6, 0],
                ['Equalize', 0.2, 8, 'Equalize', 0.6, 4],
                ['Color', 0.9, 9, 'Equalize', 0.6, 6],
                ['AutoContrast', 0.8, 4, 'Solarize', 0.2, 8],
                ['Brightness', 0.1, 3, 'Color', 0.7, 0],
                ['Solarize', 0.4, 5, 'AutoContrast', 0.9, 3],
                ['TranslateY', 0.9, 9, 'TranslateY', 0.7, 9],
                ['AutoContrast', 0.9, 2, 'Solarize', 0.8, 3],
                ['Equalize', 0.8, 8, 'Invert', 0.1, 3],
                ['TranslateY', 0.7, 9, 'AutoContrast', 0.9, 1],
            ]

    def flow(self, x, y=None, batch_size=32, shuffle=True, sample_weight=None,
             seed=None, save_to_dir=None, save_prefix='', save_format='png', subset=None):
        batches = self.datagen.flow(x, y, batch_size, shuffle, sample_weight,
                               seed, save_to_dir, save_prefix, save_format, subset)

        while True:
            x_batch, y_batch = next(batches)

            if self.args.cutout:
                for i in range(x_batch.shape[0]):
                    x_batch[i] = cutout(x_batch[i])

            if self.args.auto_augment:
                x_batch = x_batch.astype('uint8')
                for i in range(x_batch.shape[0]):
                    x_batch[i] = apply_policy(x_batch[i], self.policies[random.randrange(len(self.policies))])

            yield x_batch, y_batch

operations = {
    'ShearX': lambda img, magnitude: shear_x(img, magnitude),
    'ShearY': lambda img, magnitude: shear_y(img, magnitude),
    'TranslateX': lambda img, magnitude: translate_x(img, magnitude),
    'TranslateY': lambda img, magnitude: translate_y(img, magnitude),
    'Rotate': lambda img, magnitude: rotate(img, magnitude),
    'AutoContrast': lambda img, magnitude: auto_contrast(img, magnitude),
    'Invert': lambda img, magnitude: invert(img, magnitude),
    'Equalize': lambda img, magnitude: equalize(img, magnitude),
    'Solarize': lambda img, magnitude: solarize(img, magnitude),
    'Posterize': lambda img, magnitude: posterize(img, magnitude),
    'Contrast': lambda img, magnitude: contrast(img, magnitude),
    'Color': lambda img, magnitude: color(img, magnitude),
    'Brightness': lambda img, magnitude: brightness(img, magnitude),
    'Sharpness': lambda img, magnitude: sharpness(img, magnitude),
    'Cutout': lambda img, magnitude: cutout(img, magnitude),
}


def apply_policy(img, policy):
    if random.random() < policy[1]:
        img = operations[policy[0]](img, policy[2])
    if random.random() < policy[4]:
        img = operations[policy[3]](img, policy[5])

    return img


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = offset_matrix @ matrix @ reset_matrix
    return transform_matrix


def shear_x(img, magnitude):
    magnitudes = np.linspace(-0.3, 0.3, 11)

    transform_matrix = np.array([[1, random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]), 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    img = np.stack([ndimage.interpolation.affine_transform(
                    img[:, :, c],
                    affine_matrix,
                    offset) for c in range(img.shape[2])], axis=2)
    return img


def shear_y(img, magnitude):
    magnitudes = np.linspace(-0.3, 0.3, 11)

    transform_matrix = np.array([[1, 0, 0],
                                 [random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]), 1, 0],
                                 [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    img = np.stack([ndimage.interpolation.affine_transform(
                    img[:, :, c],
                    affine_matrix,
                    offset) for c in range(img.shape[2])], axis=2)
    return img


def translate_x(img, magnitude):
    magnitudes = np.linspace(-150/331, 150/331, 11)

    transform_matrix = np.array([[1, 0, 0],
                                 [0, 1, img.shape[1]*random.uniform(magnitudes[magnitude], magnitudes[magnitude+1])],
                                 [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    img = np.stack([ndimage.interpolation.affine_transform(
                    img[:, :, c],
                    affine_matrix,
                    offset) for c in range(img.shape[2])], axis=2)
    return img


def translate_y(img, magnitude):
    magnitudes = np.linspace(-150/331, 150/331, 11)

    transform_matrix = np.array([[1, 0, img.shape[0]*random.uniform(magnitudes[magnitude], magnitudes[magnitude+1])],
                                 [0, 1, 0],
                                 [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    img = np.stack([ndimage.interpolation.affine_transform(
                    img[:, :, c],
                    affine_matrix,
                    offset) for c in range(img.shape[2])], axis=2)
    return img


def rotate(img, magnitude):
    magnitudes = np.linspace(-30, 30, 11)

    theta = np.deg2rad(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
    transform_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                 [np.sin(theta), np.cos(theta), 0],
                                 [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    img = np.stack([ndimage.interpolation.affine_transform(
                    img[:, :, c],
                    affine_matrix,
                    offset) for c in range(img.shape[2])], axis=2)
    return img


def auto_contrast(img, magnitude):
    img = Image.fromarray(img)
    img = ImageOps.autocontrast(img)
    img = np.array(img)
    return img


def invert(img, magnitude):
    img = Image.fromarray(img)
    img = ImageOps.invert(img)
    img = np.array(img)
    return img


def equalize(img, magnitude):
    img = Image.fromarray(img)
    img = ImageOps.equalize(img)
    img = np.array(img)
    return img


def solarize(img, magnitude):
    magnitudes = np.linspace(0, 256, 11)

    img = Image.fromarray(img)
    img = ImageOps.solarize(img, random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
    img = np.array(img)
    return img


def posterize(img, magnitude):
    magnitudes = np.linspace(4, 8, 11)

    img = Image.fromarray(img)
    img = ImageOps.posterize(img, int(round(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))))
    img = np.array(img)
    return img


def contrast(img, magnitude):
    magnitudes = np.linspace(0.1, 1.9, 11)

    img = Image.fromarray(img)
    img = ImageEnhance.Contrast(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
    img = np.array(img)
    return img


def color(img, magnitude):
    magnitudes = np.linspace(0.1, 1.9, 11)

    img = Image.fromarray(img)
    img = ImageEnhance.Color(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
    img = np.array(img)
    return img


def brightness(img, magnitude):
    magnitudes = np.linspace(0.1, 1.9, 11)

    img = Image.fromarray(img)
    img = ImageEnhance.Brightness(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
    img = np.array(img)
    return img


def sharpness(img, magnitude):
    magnitudes = np.linspace(0.1, 1.9, 11)

    img = Image.fromarray(img)
    img = ImageEnhance.Sharpness(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
    img = np.array(img)
    return img


def cutout(org_img, magnitude=None):
    magnitudes = np.linspace(0, 60/331, 11)

    img = np.copy(org_img)
    mask_val = img.mean()

    if magnitude is None:
        mask_size = 16
    else:
        mask_size = int(round(img.shape[0]*random.uniform(magnitudes[magnitude], magnitudes[magnitude+1])))
    top = np.random.randint(0 - mask_size//2, img.shape[0] - mask_size)
    left = np.random.randint(0 - mask_size//2, img.shape[1] - mask_size)
    bottom = top + mask_size
    right = left + mask_size

    if top < 0:
        top = 0
    if left < 0:
        left = 0

    img[top:bottom, left:right, :].fill(mask_val)

    return img