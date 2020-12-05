import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf  #
import numpy as np
import os, sys
import data_input_helper as data_helpers
import jieba

import tensorflow.compat.v1 as tf  #
tf.disable_v2_behavior()

##数据参数
tf.compat.v1.flags.DEFINE_string("w2v_file", "./data/word2vec.bin", "w2v_file path")
##评估模型参数
tf.compat.v1.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.compat.v1.flags.DEFINE_string("checkpoint_dir", "./runs/checkpoints/", "Checkpoint directory from training run")
##tf.ConfigProto参数
tf.compat.v1.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.compat.v1.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.compat.v1.flags.FLAGS
FLAGS.flag_values_dict()  #

class RefuseClassification():  ##垃圾分类

    def __init__(self):    
        self.w2v_wr = data_helpers.w2v_wrapper(FLAGS.w2v_file)  ##加载词向量
        self.init_model()  ##初始化模型
        self.refuse_classification_map = {0: '可回收垃圾', 1: '有害垃圾', 2: '湿垃圾', 3: '干垃圾'}    
        
    def deal_data(self, text, max_document_length = 10):  ##处理数据获取标签（类别）0123      
        words = jieba.cut(text)  ##结巴分词
        x_text = [' '.join(words)]
        x = data_helpers.get_text_idx(x_text, self.w2v_wr.model.vocab_hash, max_document_length)
        return x
        
    def init_model(self):  ##初始化模型
        ##checkpoint_dir-->D:/.../runs/checkpoints/"  训练模型
        checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)  ##自动寻找最新的checkpoint
        graph = tf.Graph()  ##实例化数据流图
        with graph.as_default():  ##定义计算图的张量和操作
            session_conf = tf.compat.v1.ConfigProto(
                              allow_soft_placement=FLAGS.allow_soft_placement,  ##True 允许自动备份 
                              log_device_placement=FLAGS.log_device_placement)  ##False 不打印设备分配日志
            self.sess = tf.compat.v1.Session(config=session_conf)  ##基本格式，执行网络，进行计算
            self.sess.as_default()
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))  ##加载`xxx.meta`中的图（节点参数）
            saver.restore(self.sess, checkpoint_file)#恢复变量，saver类训练完以checkpoints文件形式保存，提取时也是从其恢复变量
            self.input_x = graph.get_operation_by_name("input_x").outputs[0]  ##获取节点操作，传入占位符参数
            self.dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            self.predictions = graph.get_operation_by_name("output/predictions").outputs[0]  ##评估  
            
    def predict(self, text):  ##预测类别0123  
        x_test = self.deal_data(text, 5)  ##处理数据获取标签（类别）0123
        predictions = self.sess.run(self.predictions, {self.input_x: x_test, self.dropout_keep_prob: 1.0})  ##计算     
        refuse_text = self.refuse_classification_map[predictions[0]]  ##根据类别标签从字典获取键值
        return refuse_text
        
if __name__ == "__main__":
    if len(sys.argv) == 2:  ##sys.argv()包含程序本身及用户输入参数的列表
        test = RefuseClassification()
        res = test.predict(sys.argv[1])  ##sys.argv[1] 参数，即图片
        print('classify:', res)
