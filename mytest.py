from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import pathlib
import numpy as np
import tensorflow as tf

from modules.models import RRDB_Model
from modules.utils import (load_yaml, set_memory_growth, imresize_np,
                           tensor2img, rgb2ycbcr, create_lr_hr_pair,
                           calculate_psnr, calculate_ssim)

flags.DEFINE_string('cfg_path', './configs/esrgan.yaml', 'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_string('img_path', '', 'path to input image')
flags.DEFINE_integer('crop_size', 256, '')
flags.DEFINE_integer('crop_num', 3, '')


def random_crop(image, crop_size=(224, 224)):
    h, w, _ = image.shape

    # 0~(400-224)の間で画像のtop, leftを決める
    top = np.random.randint(0, h - crop_size[0])
    left = np.random.randint(0, w - crop_size[1])

    # top, leftから画像のサイズである224を足して、bottomとrightを決める
    bottom = top + crop_size[0]
    right = left + crop_size[1]

    # 決めたtop, bottom, left, rightを使って画像を抜き出す
    image = image[top:bottom, left:right, :]
    return image


def save_image(bic_img, sr_img):
    result_img_path = f'./Bic_SR_' + os.path.basename(FLAGS.img_path)
    print("[*] write the result image {}".format(result_img_path))
    results_img = np.concatenate((bic_img, sr_img), 1)
    cv2.imwrite(result_img_path, results_img)


def main(_argv):
    # init
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg = load_yaml(FLAGS.cfg_path)

    # define network
    model = RRDB_Model(None, cfg['ch_size'], cfg['network_G'])

    # load checkpoint
    checkpoint_dir = './checkpoints/' + cfg['sub_name']
    checkpoint = tf.train.Checkpoint(model=model)
    if tf.train.latest_checkpoint(checkpoint_dir):
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        print("[*] load ckpt from {}.".format(
            tf.train.latest_checkpoint(checkpoint_dir)))
    else:
        print("[*] Cannot find ckpt from {}.".format(checkpoint_dir))
        exit()

    # evaluation
    if FLAGS.img_path:
        print("[*] Processing on single image {}".format(FLAGS.img_path))
        src_lr_img = cv2.imread(FLAGS.img_path)
        h, w, _ = src_lr_img.shape

        if h < FLAGS.crop_size or w < FLAGS.crop_size:
            sr_img = tensor2img(model(src_lr_img[np.newaxis, :] / 255))
            bic_img = imresize_np(src_lr_img, cfg['scale']).astype(np.uint8)
            save_image(bic_img, sr_img, 0)

        else:
            sr_imgs = None
            bic_imgs = None
            for i in range(FLAGS.crop_num):
                lr_img = random_crop(src_lr_img,
                                     (FLAGS.crop_size, FLAGS.crop_size))

                sr_img = tensor2img(model(lr_img[np.newaxis, :] / 255))
                bic_img = imresize_np(lr_img, cfg['scale']).astype(np.uint8)

                if sr_imgs is None:
                    sr_imgs = sr_img
                    bic_imgs = bic_img
                else:
                    sr_imgs = np.concatenate((sr_imgs, sr_img), 0)
                    bic_imgs = np.concatenate((bic_imgs, bic_img), 0)

            save_image(bic_img, sr_img)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
