import warnings
warnings.filterwarnings('ignore')

import numpy as np
import os, sys
sys.path.append('textcnn')
from textcnn.predict import RefuseClassification
from classify_image import *


class RafuseRecognize():
    
    def __init__(self):
        
        self.refuse_classification = RefuseClassification()  ##调用predict.py的垃圾分类函数
        self.init_classify_image_model()  ##初始化分类图形模型
        self.node_lookup = NodeLookup(uid_chinese_lookup_path='./data/imagenet_2012_challenge_label_chinese_map.pbtxt', 
                                model_dir = '/tmp/imagenet')  ##调用classify_image.py的函数，将节点ID转化为可读标签 
        
    def init_classify_image_model(self):  ##初始化分类图形模型
        
        create_graph('/tmp/imagenet')
        self.sess = tf.Session()  ##执行网络，进行计算
        self.softmax_tensor = self.sess.graph.get_tensor_by_name('softmax:0')  ##获取图形中的tensor名       
        
    def recognize_image(self, image_data):  ##识别图片
        
        predictions = self.sess.run(self.softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})  ##计算tensor
        predictions = np.squeeze(predictions)  ##从矩阵中去掉维度为1的

        top_k = predictions.argsort()[-5:][::-1]  ##列表升序排列，取最后5个数据并从后往前输出
        result_list = []
        for node_id in top_k:
            human_string = self.node_lookup.id_to_string(node_id)  ##将节点ID转化为可读标签 
            human_string_1 = ''.join(list(set(human_string.replace('，', ',').split(','))))
            classification = self.refuse_classification.predict(human_string_1)  ##预测类别`0123`
            result_list.append('%s  =>  %s' % (human_string, classification))
            
        return '\n'.join(result_list)
        

if __name__ == "__main__":
    if len(sys.argv) == 2:  ##sys.argv()包含程序本身及用户输入参数的列表
        test = RafuseRecognize()
        image_data = tf.gfile.FastGFile(sys.argv[1], 'rb').read()  ##实现对图片的读取，rb为非UTF-8编码
        res = test.recognize_image(image_data)  ##识别图片
        print(res)
        print('classify:\n%s' %(res))
