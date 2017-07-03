import sys
import tensorflow as tf
from PIL import Image, ImageFilter

def predictint(imvalue):

	# Parameters
	x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
	y_ = tf.placeholder(tf.float32, shape=[None, 10])

	def weight_variable(shape):
	  initial = tf.truncated_normal(shape, stddev=0.1)
	  return tf.Variable(initial)

	def bias_variable(shape):
	  initial = tf.constant(0.1, shape=shape)
	  return tf.Variable(initial)

	def conv2d(x, W):
	  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

	def max_pool_2x2(x):
	  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
	                        strides=[1, 2, 2, 1], padding='SAME')

	x_image = tf.reshape(x, [-1,28,28,1])

	W_conv1 = weight_variable([5, 5, 1, 32])
	b_conv1 = bias_variable([32])

	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	W_fc1 = weight_variable([7 * 7 * 64, 1024])
	b_fc1 = bias_variable([1024])

	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	W_fc2 = weight_variable([1024, 10])
	b_fc2 = bias_variable([10])

	y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2, name="softmax")

	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	init = tf.global_variables_initializer()
	keep_prob = tf.placeholder(tf.float32)
	# For writing training checkpoints and reading them back in.
	saver = tf.train.Saver()

	
	with tf.Session() as sess:
    		sess.run(init)
        	saver.restore(sess, "model.ckpt")
        #print ("Model restored.")
       
        	prediction=prediction=tf.argmax(y_conv,1)
        	return prediction.eval(feed_dict={x: [imvalue],keep_prob: 1.0}, session=sess)

def imageprepare(argv):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255)) #creates white canvas of 28x28 pixels
    
    if width > height: #check which dimension is bigger
        #Width is bigger. Width becomes 20 pixels.
        nheight = int(round((28.0/width*height),0)) #resize height according to ratio width
        print("nheight")
        print(nheight)
        if (nheight == 0): #rare case but minimum is 1 pixel
            print("nheight == 0")
            nheight = 1  
        # resize and sharpen
        img = im.resize((28,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight)/2),0)) #caculate horizontal pozition
        newImage.paste(img, (0, 0)) #paste resized image on white canvas
    else:
        #Height is bigger. Heigth becomes 20 pixels. 
        nwidth = int(round((28.0/height*width),0)) #resize width according to ratio height
        print("nwidth")
        print(nwidth)
        if (nwidth == 0): #rare case but minimum is 1 pixel
            nwidth = 1
         # resize and sharpen
        img = im.resize((nwidth,28), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth)/2),0)) #caculate vertical pozition
        newImage.paste(img, (0, 0)) #paste resized image on white canvas
    newImage.show()
    
    #newImage.save("sample.png")

    tv = list(newImage.getdata()) #get pixel values
    
    #normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [ (255-x)*1.0/255.0 for x in tv] 
    #print(tva)
    #print(tv)
	
    return tva
    #print(tva)

def main(argv):
    """
    Main function.
    """
    imvalue = imageprepare(argv)
    predint = predictint(imvalue)
    print (predint[0]) #first value in list
    
if __name__ == "__main__":
    main(sys.argv[1])