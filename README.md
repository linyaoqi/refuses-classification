# repository
# 本次项目基于tensorflow 2.0版本，可供2.0版本以上朋友参考。

# 快速搭建垃圾分类模型

# 下载模型
    python classify_image.py    
    
# 测试模型
    python classify_image.py --image_file ./img/2.png         
    
# imagenet类别映射表   
    - 英文对照表   
        ./data/imagenet_synset_to_human_label_map.txt
    
    - 中文对照表    
        ./data/imagenet_2012_challenge_label_chinese_map.pbtxt
    
    - id对照表        
        ./data/imagenet_2012_challenge_label_map_proto.pbtxt                            

# 垃圾分类映射
    - 数据标注       
        训练数据：./data/train_data.txt
        测试数据：./data/vilid_data.txt
        
    - 模型    
        TextCNN
        
    - 详解    
        - 模型训练
            python train.py    
            训练数据：./data/train_data.txt        
    
        - 模型评估    
            python eval.py     
            测试数据：./data/vilid_data.txt

        - 单句测试
            python predict.py '猪肉饺子'    
            输出结果：classify: 湿垃圾
                        
# 垃圾分类识别
    - 识别
        python refuse.py img/2.png        
        输出结果：
            移动电话手机  =>  可回收垃圾
            iPod  =>  湿垃圾
            笔记本笔记本电脑  =>  可回收垃圾
            调制解调器  =>  湿垃圾
            手持电脑手持微电脑  =>  可回收垃圾
