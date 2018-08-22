from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from collections import namedtuple

from module import *
from utils import *

D_LOSS_THRESHOLD = 0.3

class cyclegan(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        self.in_size = args.load_size
        self.image_size = args.fine_size
        self.input_c_dim = args.input_nc
        self.output_c_dim = args.output_nc
        self.L1_lambda = args.L1_lambda
        self.dataset_dir = args.dataset_dir

        self.discriminator = discriminator
        self.generator = generator_unet
        self.criterionGAN = sce_criterion

        OPTIONS = namedtuple('OPTIONS', 'batch_size in_size image_size \
                              gf_dim df_dim output_c_dim is_training')
        self.options = OPTIONS._make((args.batch_size, args.load_size, args.fine_size,
                                      args.ngf, args.ndf, args.output_nc,
                                      args.phase == 'train'))

        self._build_model()
        self.saver = tf.train.Saver()
        self.pool = ImagePool(args.max_size)

    def _build_model(self):
        self.real_data = tf.placeholder(tf.float32,
                                        [self.in_size, self.in_size,
                                         self.input_c_dim + self.output_c_dim],
                                        name='real_A_and_B_images')
        with tf.device("/device:CPU:0"):
            self.real_A = self.random_tiles(self.real_data[:, :, :self.input_c_dim])
            self.real_B = self.random_tiles(self.real_data[:, :, self.input_c_dim:self.input_c_dim + self.output_c_dim])

        with tf.device("/device:GPU:1"):
            self.fake_B = self.generator(self.real_A, self.options, False, name="generatorA2B")
            # With the true U-Net model we lose edge pixels with each transform.
            # As a result, the fakes are smaller than the reals, and after cycling
            # back become smaller again.
            self.fake_width = int(self.fake_B.shape[1])
            self.fake_height = int(self.fake_B.shape[2])
            self.fake_A_ = self.generator(self.fake_B, self.options, False, name="generatorB2A")
            self.cycled_width = int(self.fake_A_.shape[1])
            self.cycled_height = int(self.fake_A_.shape[2])

        with tf.device("/device:GPU:2"):
            self.fake_A = self.generator(self.real_B, self.options, True, name="generatorB2A")
            self.fake_B_ = self.generator(self.fake_A, self.options, True, name="generatorA2B")

        with tf.device("/device:GPU:3"):
            self.DB_fake = self.discriminator(self.fake_B, self.options, reuse=False, name="discriminatorB")
            self.DA_fake = self.discriminator(self.fake_A, self.options, reuse=False, name="discriminatorA")
            self.g_loss_a2b = self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake)) \
                + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
                + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)
            self.g_loss_b2a = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) \
                + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
                + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)
            self.g_loss = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) \
                + self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake)) \
                + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
                + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)

            self.fake_A_sample = tf.placeholder(tf.float32,
                                                [None, self.fake_width, self.fake_height,
                                                 self.input_c_dim], name='fake_A_sample')
            self.fake_B_sample = tf.placeholder(tf.float32,
                                                [None, self.fake_width, self.fake_height,
                                                 self.output_c_dim], name='fake_B_sample')
            # Discriminator input size is fake_width x fake_height - crop reals
            self.DB_real = self.discriminator(centre_crop(self.real_B, self.fake_width, self.fake_height), self.options, reuse=True, name="discriminatorB")
            self.DA_real = self.discriminator(centre_crop(self.real_A, self.fake_width, self.fake_height), self.options, reuse=True, name="discriminatorA")
            self.DB_fake_sample = self.discriminator(self.fake_B_sample, self.options, reuse=True, name="discriminatorB")
            self.DA_fake_sample = self.discriminator(self.fake_A_sample, self.options, reuse=True, name="discriminatorA")

            self.db_loss_real = self.criterionGAN(self.DB_real, tf.ones_like(self.DB_real))
            self.db_loss_fake = self.criterionGAN(self.DB_fake_sample, tf.zeros_like(self.DB_fake_sample))
            self.db_loss = (self.db_loss_real + self.db_loss_fake) / 2
            self.da_loss_real = self.criterionGAN(self.DA_real, tf.ones_like(self.DA_real))
            self.da_loss_fake = self.criterionGAN(self.DA_fake_sample, tf.zeros_like(self.DA_fake_sample))
            self.da_loss = (self.da_loss_real + self.da_loss_fake) / 2
            self.d_loss = self.da_loss + self.db_loss

            self.g_loss_a2b_sum = tf.summary.scalar("g_loss_a2b", self.g_loss_a2b)
            self.g_loss_b2a_sum = tf.summary.scalar("g_loss_b2a", self.g_loss_b2a)
            self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
            self.g_sum = tf.summary.merge([self.g_loss_a2b_sum, self.g_loss_b2a_sum, self.g_loss_sum])
            self.db_loss_sum = tf.summary.scalar("db_loss", self.db_loss)
            self.da_loss_sum = tf.summary.scalar("da_loss", self.da_loss)
            self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
            self.db_loss_real_sum = tf.summary.scalar("db_loss_real", self.db_loss_real)
            self.db_loss_fake_sum = tf.summary.scalar("db_loss_fake", self.db_loss_fake)
            self.da_loss_real_sum = tf.summary.scalar("da_loss_real", self.da_loss_real)
            self.da_loss_fake_sum = tf.summary.scalar("da_loss_fake", self.da_loss_fake)
            self.d_sum = tf.summary.merge(
                [self.da_loss_sum, self.da_loss_real_sum, self.da_loss_fake_sum,
                 self.db_loss_sum, self.db_loss_real_sum, self.db_loss_fake_sum,
                 self.d_loss_sum]
            )

        self.test_A = tf.placeholder(tf.float32,
                                     [self.in_size, self.in_size,
                                      self.input_c_dim], name='test_A')
        self.test_B = tf.placeholder(tf.float32,
                                     [self.in_size, self.in_size,
                                      self.output_c_dim], name='test_B')
        self.testB = self.generator(tf.expand_dims(self.test_A, 0), self.options, True, name="generatorA2B")
        self.testA = self.generator(tf.expand_dims(self.test_B, 0), self.options, True, name="generatorB2A")

        self.real_A_summary = tf.summary.image("realA", (tf.expand_dims(self.test_A, 0)+1.)/2.)
        self.fake_A_summary = tf.summary.image("fakeA", (self.testA+1.)/2.)
        self.real_B_summary = tf.summary.image("realB", (tf.expand_dims(self.test_B, 0)+1.)/2.)
        self.fake_B_summary = tf.summary.image("fakeB", (self.testB+1.)/2.)
        self.summary_images = tf.summary.merge(
            [self.real_A_summary, self.fake_A_summary, self.real_B_summary, self.fake_B_summary]
        )
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        for var in tf.trainable_variables():
            print(var.name + ": " + str(var.shape))

    def dynamic_transform(self, image, direction='AtoB'):
        print("Image dimensions: %s" % (str(image.shape),))
        generatorName = 'generatorA2B' if direction == 'AtoB' else 'generatorB2A'
        inputTensor = tf.placeholder(tf.float32, image.shape)
        print("Input tensor dimensions: %s" % (str(inputTensor.shape),))
        outputTensor = self.generator(inputTensor, self.options, True, name=generatorName)
        print("Output tensor dimensions: %s" % (str(outputTensor.shape),))
        return (inputTensor, outputTensor)

    def random_tiles(self, image):
        tiles = []
        for i in range(self.batch_size):
            cropped = tf.random_crop(image, [self.image_size, self.image_size, image.shape[2]])
            flipped = tf.image.random_flip_left_right(cropped)
            tiles.append(flipped)
        return tf.stack(tiles)

    def train(self, args):
        """Train cyclegan"""
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()

        if args.continue_train:
            counter = self.load(args.checkpoint_dir)
            if counter > 1:
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
        dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainA'))
        dataB = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainB'))
        batch_idxs = min(min(len(dataA), len(dataB)), args.train_size)
        start_epoch = counter // batch_idxs

        for epoch in range(start_epoch, args.epoch):
            dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainA'))
            dataB = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainB'))
            np.random.shuffle(dataA)
            np.random.shuffle(dataB)
            batch_idxs = min(min(len(dataA), len(dataB)), args.train_size)
            lr = args.lr if epoch < args.epoch_step else args.lr*(args.epoch-epoch)/(args.epoch-args.epoch_step)
            last_d_loss = 1

            for idx in range(0, batch_idxs):
                batch_files = (dataA[idx], dataB[idx])
                images = load_train_data(batch_files, args.load_size, args.fine_size)
                images = np.array(images).astype(np.float32)

                # Update G network and record fake outputs
                fake_A, fake_B, _, summary_str = self.sess.run(
                    [self.fake_A, self.fake_B, self.g_optim, self.g_sum],
                    feed_dict={self.real_data: images, self.lr: lr})
                self.writer.add_summary(summary_str, counter)
                [fake_A, fake_B] = self.pool([fake_A, fake_B])

                # Update D network
                if last_d_loss >= D_LOSS_THRESHOLD:
                    _, run_d_loss, summary_str = self.sess.run(
                        [self.d_optim, self.d_loss, self.d_sum],
                        feed_dict={self.real_data: images,
                                   self.fake_A_sample: fake_A,
                                   self.fake_B_sample: fake_B,
                                   self.lr: lr})
                    last_d_loss = run_d_loss
                else:
                    last_d_loss *= 2
                self.writer.add_summary(summary_str, counter)

                counter += 1
                print(("Epoch: [%2d] [%4d/%4d] time: %4.4f" % (
                    epoch, idx, batch_idxs, time.time() - start_time)))

                if np.mod(counter, args.print_freq) == 1:
                    self.sample_model(args.sample_dir, epoch, idx, counter, args.load_size, args.fine_size)

                if np.mod(counter, args.save_freq) == 2:
                    self.save(args.checkpoint_dir, counter)

    def save(self, checkpoint_dir, step):
        model_name = "cyclegan.model"
        model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            return counter
        else:
            return 1

    def sample_model(self, sample_dir, epoch, idx, counter, load_size=286, fine_size=256):
        dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))
        dataB = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testB'))
        np.random.shuffle(dataA)
        np.random.shuffle(dataB)
        sample_a = load_test_data(dataA[0], load_size)
        sample_b = load_test_data(dataB[0], load_size)
        #sample_images = np.array(sample_images).astype(np.float32)

        fake_A, fake_B, summary_images = self.sess.run(
            [self.testA, self.testB, self.summary_images],
            feed_dict={self.test_A: sample_a, self.test_B: sample_b}
        )
        self.writer.add_summary(summary_images, counter)
        save_images(fake_A, [self.batch_size, 1],
                    './{}/A_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))
        save_images(fake_B, [self.batch_size, 1],
                    './{}/B_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))

    def test(self, args):
        """Test cyclegan"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        if args.which_direction == 'AtoB':
            sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))
        elif args.which_direction == 'BtoA':
            sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testB'))
        else:
            raise Exception('--which_direction must be AtoB or BtoA')

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(args.test_dir, '{0}_index.html'.format(args.which_direction))
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        for sample_file in sample_files:
            print('Processing image: ' + sample_file)
            sample_image = [load_test_data(sample_file, None)]
            sample_image = np.array(sample_image).astype(np.float32)
            image_path = os.path.join(args.test_dir,
                                      '{0}_{1}'.format(args.which_direction, os.path.basename(sample_file)))
            in_var, out_var = self.dynamic_transform(sample_image, args.which_direction)
            fake_img = self.sess.run(out_var, feed_dict={in_var: sample_image})
            save_images(fake_img, [1, 1], image_path)
            index.write("<td>%s</td>" % os.path.basename(image_path))
            index.write("<td><img src='%s'></td>" % (sample_file if os.path.isabs(sample_file) else (
                '..' + os.path.sep + sample_file)))
            index.write("<td><img src='%s'></td>" % (image_path if os.path.isabs(image_path) else (
                '..' + os.path.sep + image_path)))
            index.write("</tr>")
        index.close()
