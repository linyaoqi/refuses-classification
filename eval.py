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
import tensorflow.compat.v1 as tf
import csv

# Parameters

# Data Parameters
tf.flags.DEFINE_string("valid_data_file", "./data/valid_data.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("w2v_file", "./data/word2vec.bin", "w2v_file path")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/checkpoints/", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
#FLAGS._parse_flags()
FLAGS.flag_values_dict()

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

def load_data(w2v_model,max_document_length = 20):  ##加载数据
    """Loads starter word-vectors and train/dev/test data."""
    print("Loading data...")
    x_text, y_test = data_helpers.load_data_and_labels(FLAGS.valid_data_file)  ##加载数据和生成标签
    y_test = np.argmax(y_test, axis=1)
    if(max_document_length == 0) :
        max_document_length = max([len(x.split(" ")) for x in x_text])
    print ('max_document_length = ' , max_document_length)
    x = data_helpers.get_text_idx(x_text,w2v_model.vocab_hash,max_document_length)##获取id，import data_input_helper as data_helpers
    return x,y_test

def eval(w2v_model):  ##评估
    ##初始化模型，类似`predict.py的init_model()`
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)  ##自动寻找最新的checkpoint
    graph = tf.Graph()  ##实例化数据流图
    with graph.as_default():  ##定义计算图的张量和操作
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,  ##True 允许自动备份 
          log_device_placement=FLAGS.log_device_placement)  ##False 不打印设备分配日志
        sess = tf.Session(config=session_conf)  ##执行网络，进行计算
        with sess.as_default():  ##加载图片节点参数并恢复变量
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            input_x = graph.get_operation_by_name("input_x").outputs[0]  ##获取节点操作，传入占位符参数
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]
            ##load_data()，类似train.py的`load_data()`           
            x_test, y_test = load_data(w2v_model, 5)  
            batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)            
            all_predictions = []  ##收集预测结果
            for x_test_batch in batches:  ##循环batch
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])
    if y_test is not None:
        correct_predictions = float(sum(all_predictions == y_test))
        print("Total number of test examples: {}".format(len(y_test)))
        print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))
    predictions_human_readable = np.column_stack(all_predictions)  ##保存评估结果
    out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
    print("Saving evaluation to {0}".format(out_path))
    with open(out_path, 'w') as f:
        csv.writer(f).writerows(predictions_human_readable)
        
if __name__ == "__main__":
    w2v_wr = data_helpers.w2v_wrapper(FLAGS.w2v_file)  ##加载词向量,`import data_input_helper as data_helpers`
    eval(w2v_wr.model)
