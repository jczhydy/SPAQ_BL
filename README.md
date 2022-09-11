# SPAQ_BL
  2020CVPR论文[Perceptual Quality Assessment of Smartphone Photography](https://openaccess.thecvf.com/content_CVPR_2020/papers/Fang_Perceptual_Quality_Assessment_of_Smartphone_Photography_CVPR_2020_paper.pdf) 中Yuming Fang, Hanwei Zhu, Yan Zeng, Kede Ma和Zhou Wang以resnet5O为Baseline,建立MT-E,MT-A,MT-S等衍生NR-IQA模型。论文作者使用自己建立的SPAQ数据集，给出了11125张自然手机拍摄图片及每张图片的MOS值（取值范围为0~100）,所有的图片以及对应的MOS可在作者给出的[github链接](https://github.com/h4nwei/SPAQ)中下载.作者提供的代码中只给出了模型的简单验证部分，此project补充了SPAQ数据库的实践落地、文章Baseline部分的网络搭建及训练，较好复现了原论文中的效果（Baseline部分此project训练得到SROCC为0.89，原论文中SROCC为0.90）

