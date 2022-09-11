# SPAQ_BL
  2020CVPR论文[Perceptual Quality Assessment of Smartphone Photography](https://openaccess.thecvf.com/content_CVPR_2020/papers/Fang_Perceptual_Quality_Assessment_of_Smartphone_Photography_CVPR_2020_paper.pdf) 中Yuming Fang, Hanwei Zhu, Yan Zeng, Kede Ma和Zhou Wang以resnet50为Baseline,建立MT-E,MT-A,MT-S等衍生NR-IQA模型。论文作者使用自己建立的SPAQ数据集，给出了11125张自然手机拍摄图片及每张图片的MOS值（取值范围为0~100）,所有的图片以及对应的MOS可在作者给出的[github链接](https://github.com/h4nwei/SPAQ)中下载.作者提供的代码中只给出了模型的简单验证部分，此project补充了SPAQ数据库的实践落地、文章Baseline部分的网络搭建及训练，较好复现了原论文中的效果（Baseline部分此project训练得到SROCC为0.89，原论文中SROCC为0.90）
## 训练步骤
1.将原数据集的前8000张作为训练集，导入train2文件夹中；8001~10000张作为训练集，导入test2文件夹中；创建traindata2.xlsx，在表格A1：A8000导入训练图片对应的MOS值；创建testdata2.xlsx，在表格A1：A2000导入测试图片对应的MOS值
2.为统一并缩减图片尺寸，通过getimage.py代码，将traindata和testdata中图片进行CenterCrop（size=1800）及resize操作，得到尺寸为512$/times$512的图片，分别导入到文件夹rand_train_4和rand_test_4中
3.运行train.py，得到预训练模型（Code中已给出）
4.test.py加载预训练模型得到最终的SROCC分数
