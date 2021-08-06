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
        lr_img = cv2.imread(FLAGS.img_path)
        h, w, _ = lr_img.shape

        if h > 480 or w > 480:
            lr_img = lr_img[:480, :480, :]

        sr_img = tensor2img(model(lr_img[np.newaxis, :] / 255))
        bic_img = imresize_np(lr_img, cfg['scale']).astype(np.uint8)

        result_img_path = './Bic_SR_HR_' + os.path.basename(FLAGS.img_path)
        print("[*] write the result image {}".format(result_img_path))
        results_img = np.concatenate((bic_img, sr_img), 1)
        cv2.imwrite(result_img_path, results_img)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
