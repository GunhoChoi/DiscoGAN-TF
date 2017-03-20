import tensorflow as tf
import matplotlib.pyplot as plt

# Hyperparameters

initializer = tf.truncated_normal_initializer(stddev=0.02)
learning_rate = 0.0002
batch_size = 256
epoch = 100000

# Read image files

shoes_filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once("./shoes/*.jpg"), capacity=200)
bags_filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once("./bags/*.jpg"), capacity=200)
image_reader = tf.WholeFileReader()

_, shoes_file = image_reader.read(shoes_filename_queue)
_, bags_file = image_reader.read(bags_filename_queue)

shoes_image = tf.image.decode_jpeg(shoes_file)
bags_image = tf.image.decode_jpeg(bags_file)

shoes_image = tf.cast(tf.reshape(shoes_image,shape=[64,64,3]),dtype=tf.float32)/255.0
bags_image = tf.cast(tf.reshape(bags_image,shape=[64,64,3]),dtype=tf.float32)/255.0

num_preprocess_threads = 1
min_queue_examples = 256
batch_shoes = tf.train.shuffle_batch([shoes_image],
                                     batch_size=batch_size,
                                     num_threads=num_preprocess_threads,
                                     capacity=min_queue_examples + 3 * batch_size,
                                     min_after_dequeue=min_queue_examples)

batch_bags = tf.train.shuffle_batch([bags_image],
                                    batch_size=batch_size,
                                    num_threads=num_preprocess_threads,
                                    capacity=min_queue_examples + 3 * batch_size,
                                    min_after_dequeue=min_queue_examples)


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def discriminator_shoes(tensor,reuse=False):
    
    with tf.variable_scope("discriminator_s"):

        conv1 = tf.contrib.layers.conv2d(inputs=tensor, num_outputs=32, kernel_size=4, stride=2, padding="SAME", \
                                        reuse=reuse, activation_fn=lrelu,weights_initializer=initializer,scope="d_conv1") # 32 x 32 x 32
        conv2 = tf.contrib.layers.conv2d(inputs=conv1, num_outputs=64, kernel_size=4, stride=2, padding="SAME", \
                                        reuse=reuse, activation_fn=lrelu,normalizer_fn=tf.contrib.layers.batch_norm,\
                                        weights_initializer=initializer,scope="d_conv2")                                  # 16 x 16 x 64
        conv3 = tf.contrib.layers.conv2d(inputs=conv2, num_outputs=128, kernel_size=4, stride=2, padding="SAME", \
                                        reuse=reuse, activation_fn=lrelu,normalizer_fn=tf.contrib.layers.batch_norm,\
                                        weights_initializer=initializer,scope="d_conv3")                                  # 8 x 8 x 128 
        conv4 = tf.contrib.layers.conv2d(inputs=conv3, num_outputs=256, kernel_size=4, stride=2, padding="SAME", \
                                        reuse=reuse, activation_fn=lrelu,normalizer_fn=tf.contrib.layers.batch_norm,\
                                        weights_initializer=initializer,scope="d_conv4")                                  # 4 x 4 x 256
        conv5 = tf.contrib.layers.conv2d(inputs=conv4, num_outputs=512, kernel_size=4, stride=2, padding="SAME", \
                                        reuse=reuse, activation_fn=lrelu,normalizer_fn=tf.contrib.layers.batch_norm,\
                                        weights_initializer=initializer,scope="d_conv5")                                  # 2 x 2 x 512
        fc1 = tf.reshape(conv5, shape=[batch_size, 2*2*512])
        fc1 = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=512,reuse=reuse, activation_fn=lrelu, \
                                                normalizer_fn=tf.contrib.layers.batch_norm, \
                                                weights_initializer=initializer,scope="d_fc1")
        fc2 = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=1, reuse=reuse, activation_fn=tf.nn.sigmoid,\
                                                weights_initializer=initializer,scope="d_fc2")

        return fc2


def discriminator_bags(tensor,reuse=False):
    
    with tf.variable_scope("discriminator_b"):

        conv1 = tf.contrib.layers.conv2d(inputs=tensor, num_outputs=32, kernel_size=4, stride=2, padding="SAME", \
                                        reuse=reuse, activation_fn=lrelu,weights_initializer=initializer,scope="d_conv1") # 32 x 32 x 32
        conv2 = tf.contrib.layers.conv2d(inputs=conv1, num_outputs=64, kernel_size=4, stride=2, padding="SAME", \
                                        reuse=reuse, activation_fn=lrelu,normalizer_fn=tf.contrib.layers.batch_norm,\
                                        weights_initializer=initializer,scope="d_conv2")                                  # 16 x 16 x 64
        conv3 = tf.contrib.layers.conv2d(inputs=conv2, num_outputs=128, kernel_size=4, stride=2, padding="SAME", \
                                        reuse=reuse, activation_fn=lrelu,normalizer_fn=tf.contrib.layers.batch_norm,\
                                        weights_initializer=initializer,scope="d_conv3")                                  # 8 x 8 x 128 
        conv4 = tf.contrib.layers.conv2d(inputs=conv3, num_outputs=256, kernel_size=4, stride=2, padding="SAME", \
                                        reuse=reuse, activation_fn=lrelu,normalizer_fn=tf.contrib.layers.batch_norm,\
                                        weights_initializer=initializer,scope="d_conv4")                                  # 4 x 4 x 256
        conv5 = tf.contrib.layers.conv2d(inputs=conv4, num_outputs=512, kernel_size=4, stride=2, padding="SAME", \
                                        reuse=reuse, activation_fn=lrelu,normalizer_fn=tf.contrib.layers.batch_norm,\
                                        weights_initializer=initializer,scope="d_conv5")                                  # 2 x 2 x 512
        fc1 = tf.reshape(conv5, shape=[batch_size, 2*2*512])
        fc1 = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=512,reuse=reuse, activation_fn=lrelu, \
                                                normalizer_fn=tf.contrib.layers.batch_norm, \
                                                weights_initializer=initializer,scope="d_fc1")
        fc2 = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=1, reuse=reuse, activation_fn=tf.nn.sigmoid,\
                                                weights_initializer=initializer,scope="d_fc2")

        return fc2


def generator_sb(image,reuse=False):

    with tf.variable_scope("generator_sb"):

        conv1 = tf.contrib.layers.conv2d(inputs=image, num_outputs=32, kernel_size=4, stride=2, padding="SAME", \
                                        reuse=reuse, activation_fn=lrelu,weights_initializer=initializer,scope="d_conv1") # 32 x 32 x 32
        conv2 = tf.contrib.layers.conv2d(inputs=conv1, num_outputs=64, kernel_size=4, stride=2, padding="SAME", \
                                        reuse=reuse, activation_fn=lrelu,normalizer_fn=tf.contrib.layers.batch_norm,\
                                        weights_initializer=initializer,scope="d_conv2")                                  # 16 x 16 x 64
        conv3 = tf.contrib.layers.conv2d(inputs=conv2, num_outputs=128, kernel_size=4, stride=2, padding="SAME", \
                                        reuse=reuse, activation_fn=lrelu,normalizer_fn=tf.contrib.layers.batch_norm,\
                                        weights_initializer=initializer,scope="d_conv3")                                  # 8 x 8 x 128 
        conv4 = tf.contrib.layers.conv2d(inputs=conv3, num_outputs=256, kernel_size=4, stride=2, padding="SAME", \
                                        reuse=reuse, activation_fn=lrelu,normalizer_fn=tf.contrib.layers.batch_norm,\
                                        weights_initializer=initializer,scope="d_conv4")                                  # 4 x 4 x 256
        
        conv_trans1 = tf.contrib.layers.conv2d(conv4, num_outputs=4*128, kernel_size=4, stride=1, padding="SAME",    \
                                        activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm, \
                                        weights_initializer=initializer,scope="g_conv1")
        conv_trans1 = tf.reshape(conv_trans1, shape=[batch_size,8,8,128])

        conv_trans2 = tf.contrib.layers.conv2d(conv_trans1, num_outputs=4*64, kernel_size=4, stride=1, padding="SAME", \
                                        activation_fn=tf.nn.relu,normalizer_fn=tf.contrib.layers.batch_norm, \
                                        weights_initializer=initializer,scope="g_conv2")
        conv_trans2 = tf.reshape(conv_trans2, shape=[batch_size,16,16,64])

        conv_trans3 = tf.contrib.layers.conv2d(conv_trans2, num_outputs=4*32, kernel_size=4, stride=1, padding="SAME",    \
                                        activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm, \
                                        weights_initializer=initializer,scope="g_conv3")
        conv_trans3 = tf.reshape(conv_trans3, shape=[batch_size,32,32,32])

        conv_trans4 = tf.contrib.layers.conv2d(conv_trans3, num_outputs=4*16, kernel_size=4, stride=1, padding="SAME", \
                                        activation_fn=tf.nn.relu,normalizer_fn=tf.contrib.layers.batch_norm, \
                                        weights_initializer=initializer,scope="g_conv4")
        conv_trans4 = tf.reshape(conv_trans4, shape=[batch_size,64,64,16])

        recon_bag = tf.contrib.layers.conv2d(conv_trans4, num_outputs=3, kernel_size=4, stride=1, padding="SAME", \
                                        activation_fn=tf.nn.relu,scope="g_conv5")

        return recon_bag


def generator_bs(image,reuse=False):

    with tf.variable_scope("generator_bs"):

        conv1 = tf.contrib.layers.conv2d(inputs=image, num_outputs=32, kernel_size=4, stride=2, padding="SAME", \
                                        reuse=reuse, activation_fn=lrelu,weights_initializer=initializer,scope="d_conv1") # 32 x 32 x 32
        conv2 = tf.contrib.layers.conv2d(inputs=conv1, num_outputs=64, kernel_size=4, stride=2, padding="SAME", \
                                        reuse=reuse, activation_fn=lrelu,normalizer_fn=tf.contrib.layers.batch_norm,\
                                        weights_initializer=initializer,scope="d_conv2")                                  # 16 x 16 x 64
        conv3 = tf.contrib.layers.conv2d(inputs=conv2, num_outputs=128, kernel_size=4, stride=2, padding="SAME", \
                                        reuse=reuse, activation_fn=lrelu,normalizer_fn=tf.contrib.layers.batch_norm,\
                                        weights_initializer=initializer,scope="d_conv3")                                  # 8 x 8 x 128 
        conv4 = tf.contrib.layers.conv2d(inputs=conv3, num_outputs=256, kernel_size=4, stride=2, padding="SAME", \
                                        reuse=reuse, activation_fn=lrelu,normalizer_fn=tf.contrib.layers.batch_norm,\
                                        weights_initializer=initializer,scope="d_conv4")                                  # 4 x 4 x 256
        
        conv_trans1 = tf.contrib.layers.conv2d(conv4, num_outputs=4*128, kernel_size=4, stride=1, padding="SAME",    \
                                        activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm, \
                                        weights_initializer=initializer,scope="g_conv1")
        conv_trans1 = tf.reshape(conv_trans1, shape=[batch_size,8,8,128])

        conv_trans2 = tf.contrib.layers.conv2d(conv_trans1, num_outputs=4*64, kernel_size=4, stride=1, padding="SAME", \
                                        activation_fn=tf.nn.relu,normalizer_fn=tf.contrib.layers.batch_norm, \
                                        weights_initializer=initializer,scope="g_conv2")
        conv_trans2 = tf.reshape(conv_trans2, shape=[batch_size,16,16,64])

        conv_trans3 = tf.contrib.layers.conv2d(conv_trans2, num_outputs=4*32, kernel_size=4, stride=1, padding="SAME",    \
                                        activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm, \
                                        weights_initializer=initializer,scope="g_conv3")
        conv_trans3 = tf.reshape(conv_trans3, shape=[batch_size,32,32,32])

        conv_trans4 = tf.contrib.layers.conv2d(conv_trans3, num_outputs=4*16, kernel_size=4, stride=1, padding="SAME", \
                                        activation_fn=tf.nn.relu,normalizer_fn=tf.contrib.layers.batch_norm, \
                                        weights_initializer=initializer,scope="g_conv4")
        conv_trans4 = tf.reshape(conv_trans4, shape=[batch_size,64,64,16])

        recon_shoe = tf.contrib.layers.conv2d(conv_trans4, num_outputs=3, kernel_size=4, stride=1, padding="SAME", \
                                        activation_fn=tf.nn.relu,scope="g_conv5")

        return recon_shoe

# Generation & Discrimination

gen_b_fake = generator_sb(batch_shoes)
gen_s_fake = generator_bs(batch_bags)

recon_s = generator_bs(gen_b_fake,reuse=True)
recon_b = generator_sb(gen_s_fake,reuse=True)

disc_s_fake = discriminator_shoes(gen_s_fake)
disc_b_fake = discriminator_bags(gen_b_fake)

disc_s_real = discriminator_shoes(batch_shoes,reuse=True)
disc_b_real = discriminator_bags(batch_bags,reuse=True)

# Loss Caculation

const_loss_s = tf.reduce_sum(tf.losses.mean_squared_error(batch_shoes,recon_s))
const_loss_b = tf.reduce_sum(tf.losses.mean_squared_error(batch_bags,recon_b))

gen_s_loss = tf.reduce_sum(tf.square(disc_s_fake-1))/2
gen_b_loss = tf.reduce_sum(tf.square(disc_b_fake-1))/2

disc_s_loss = tf.reduce_sum(tf.square(disc_s_real-1) + tf.square(disc_s_fake))/2
disc_b_loss = tf.reduce_sum(tf.square(disc_b_real-1) + tf.square(disc_b_fake))/2

gen_loss = const_loss_s + const_loss_b + gen_s_loss + gen_b_loss
disc_loss = disc_s_loss + disc_b_loss

gen_sb_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator_sb")
gen_bs_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator_bs")
dis_s_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator_s") 
dis_b_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator_b") 

d_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
g_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

d_grads = d_optimizer.compute_gradients(disc_loss,dis_s_variables + dis_b_variables) #Only update the weights for the discriminator network.
g_grads = g_optimizer.compute_gradients(gen_loss,gen_sb_variables + gen_bs_variables) #Only update the weights for the generator network.

update_D = d_optimizer.apply_gradients(d_grads)
update_G = g_optimizer.apply_gradients(g_grads)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    try:
        saver.restore(sess=sess, save_path="./model/model.ckpt")
        print("\n--------model restored--------\n")
    except:
        print("\n--------model Not restored--------\n")
        pass

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    for i in range(epoch):
	
        for j in range(5):
            
                _ = sess.run([update_G])
    
        _, _, g_loss, d_loss, reconed_b, reconed_s, fake_s, fake_b,s_image, b_image  = sess.run([update_G,update_D, 
                                                                                                gen_loss, disc_loss,
                                                                                                recon_b,recon_s,
                                                                                                gen_s_fake,gen_b_fake,
                                                                                                batch_shoes,batch_bags])

        if i % 100 == 0:
            saver.save(sess, './model/model.ckpt')
            print("{}th iter gen loss:{} disc loss:{}".format(i,g_loss/batch_size, d_loss/batch_size))
            plt.imsave("./result/{}th_recon_bag.png".format(i),reconed_b[0])
            plt.imsave("./result/{}th_recon_shoe.png".format(i),reconed_s[0])
            plt.imsave("./result/{}th_origin_shoe.png".format(i),s_image[0])
            plt.imsave("./result/{}th_origin_bag.png".format(i),b_image[0])
            plt.imsave("./result/{}th_gen_shoe.png".format(i),fake_s[0])
            plt.imsave("./result/{}th_gen_bag.png".format(i),fake_b[0])
    

    coord.request_stop()
    coord.join(threads)
