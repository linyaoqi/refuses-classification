import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_input_helper as data_helpers
from text_cnn import TextCNN
import math
#from tensorflow.contrib import learn
import tensorflow.compat.v1 as tf

##数据加载参数
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("train_data_file", "./data/train_data.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("train_label_data_file", "", "Data source for the label data.")
tf.flags.DEFINE_string("w2v_file", "./data/word2vec.bin", "w2v_file path")
##模型参数
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "2, 3, 4", "Comma-separated filter sizes (default: '3, 4, 5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
##训练参数
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 500, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 3, "Number of checkpoints to store (default: 5)")
##tf.ConfigProto配置Session参数
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

def load_data(w2v_model):  ##加载数据
    print("Loading data...")  ##import data_input_helper as data_helpers
    x_text, y = data_helpers.load_data_and_labels(FLAGS.train_data_file)  ##加载数据和生成标签
    max_document_length = max([len(x.split(" ")) for x in x_text])
    print ('len(x) = ', len(x_text), ' ', len(y))  ##x_text长度
    print(' max_document_length = ', max_document_length)  ##x_text中最大长度的元素
    x = []
    vocab_size = 0
    if(w2v_model is None):  ##如果模型存在
      vocab_processor = tf.contrib.preprocessing.VocabularyProcessor(max_document_length)
      x = np.array(list(vocab_processor.fit_transform(x_text)))
      vocab_size = len(vocab_processor.vocabulary_)
      vocab_processor.save("vocab.txt")
      print( 'save vocab.txt')
    else:  ##import data_input_helper as data_helpers
      x = data_helpers.get_text_idx(x_text, w2v_model.vocab_hash, max_document_length)  ##获取id
      vocab_size = len(w2v_model.vocab_hash)
      print('use w2v .bin')
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))  ##随机排序
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    return x_train, x_dev, y_train, y_dev, vocab_size
def train(w2v_model):  ##训练
    x_train, x_dev, y_train, y_dev, vocab_size= load_data(w2v_model)  ##加载数据
    with tf.Graph().as_default():  ##定义计算图的张量和操作
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,  ##True 允许自动备份 
          log_device_placement=FLAGS.log_device_placement)  ##False 不打印设备分配日志
        sess = tf.Session(config=session_conf)  ##执行网络，进行计算
        with sess.as_default():  ##创建默认会话
            cnn = TextCNN(  ##from text_cnn import TextCNN，一个文本分类的CNN
                w2v_model, 
                sequence_length=x_train.shape[1], 
                num_classes=y_train.shape[1], 
                vocab_size=vocab_size, 
                embedding_size=FLAGS.embedding_dim, 
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(", "))), 
                num_filters=FLAGS.num_filters, 
                l2_reg_lambda=FLAGS.l2_reg_lambda)
            ##定义训练程序
            global_step = tf.Variable(0, name="global_step", trainable=False)  ##创建新变量
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)  ##计算梯度
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)  ##更新梯度            
            grad_summaries = []  ##跟踪渐变值和稀疏度（可选）
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)
            ##输出模型和摘要目录
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs"))  ##返回绝对路径
            print("Writing to {}\n".format(out_dir))           
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)            
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)#Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
            #Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):  ##如果不存在checkpoint_dir
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
            sess.run(tf.global_variables_initializer())  ##初始化变量            
            def train_step(x_batch, y_batch):  ##单独的训练步骤            
                feed_dict = {
                  cnn.input_x: x_batch, 
                  cnn.input_y: y_batch, 
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy], 
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)
            def dev_step(x_batch, y_batch, writer=None):  ##评估开发集上的模型      
                feed_dict = {
                  cnn.input_x: x_batch, 
                  cnn.input_y: y_batch, 
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy], 
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)                       
            def dev_test():  ##import data_input_helper as data_helpers                       
                batches_dev = data_helpers.batch_iter(list(zip(x_dev, y_dev)), FLAGS.batch_size, 1)
                for batch_dev in batches_dev:
                    x_batch_dev, y_batch_dev = zip(*batch_dev)
                    dev_step(x_batch_dev, y_batch_dev, writer=dev_summary_writer)
            batches = data_helpers.batch_iter(  ##import data_input_helper as data_helpers
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            for batch in batches:  ##循环训练batch
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:  ##训练参数
                    print("\nEvaluation:")
                    dev_test()
                if current_step % FLAGS.checkpoint_every == 0:  ##训练参数
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
if __name__ == "__main__":  
    w2v_wr = data_helpers.w2v_wrapper(FLAGS.w2v_file)  ##加载词向量,`import data_input_helper as data_helpers`
    train(w2v_wr.model)
