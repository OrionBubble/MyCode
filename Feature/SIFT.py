from scipy import cluster

from Defines import *


def Sift(dataset):
    """提取sift特征，每张图片特征维度为[x, 128]"""
    # List where all the descriptors are stored
    sift_list = []
    fail_count = 0
    print('1、开始进行所有图片的sift特征运算')
    detector = cv2.xfeatures2d.SIFT_create()
    for image in tqdm(dataset):
        # des:[x, 128]
        kp, sift = detector.detectAndCompute(image, None)  # des是描述子 sift是关键点特征，每张图关键点不一样多，每个关键点是128维
        if sift is None:
            fail_count += 1
            sift = sift_list[len(sift_list) - 1]
        sift_list.append(sift)
    print(f' *发现{fail_count}张图片未计算出sift特征*')
    return sift_list


def Kmeans(sift_list, WOB_size):
    print('2.1、开始堆叠所有sift特征')
    descriptors = numpy.vstack(sift_list)
    print(f'shape变化：({len(sift_list)}, x, 128) -> {descriptors.shape}')
    print("2.2、开始K聚类: 聚类中心个数%d, 数据点个数%d （这个过程耗时最长）" % (WOB_size, descriptors.shape[0]))
    cluster_dict, _ = cluster.vq.kmeans(descriptors, WOB_size, iter=1)
    return cluster_dict


def BOW(sift_list, BOW_size, cluster_dict):
    im_features = numpy.zeros((len(sift_list), BOW_size), "float32")
    for i in range(len(sift_list)):
        words, _ = cluster.vq.vq(sift_list[i], cluster_dict)
        for w in words:
            im_features[i][w] += 1

    TFw = numpy.sum((im_features > 0) * 1, axis=0)
    IDFw = numpy.array(numpy.log((len(sift_list) + 1) / (TFw + 1)), 'float32')
    im_features = im_features * IDFw
    im_features = preprocessing.normalize(im_features, norm='l2')
    return im_features


# 获得特征并保存
#######################################################
def Train_Sift(train_data, train_label, test_data):
    # 将train和test集合并到一起进行特征提取（train, test）
    all_image = numpy.concatenate((train_data, test_data))
    sift_list = Sift(all_image)  # 提取sift特征
    cluster_dict = Kmeans(sift_list, SIFT_BOW_size)  # K聚类获得词典
    all_features = BOW(sift_list, SIFT_BOW_size, cluster_dict)  # 通过词袋模型计算图片所含各单词频率数据
    # 保存时，将train和test集合分开
    joblib.dump((all_features[:len(train_data)], train_label, all_features[len(train_data):]),
                f'res/sift_features_bow={SIFT_BOW_size}.pkl')


if __name__ == '__main__':
    # is_preload = False
    if not is_preload:
        train_data, train_label, test_data = Data_Load(path)
        Train_Sift(train_data, train_label, test_data)
    # 使用Sift特征进行预测
    train_data, train_label, test_data = joblib.load(f'res/sift_features_bow={SIFT_BOW_size}.pkl')
    train_X, val_X, train_Y, val_Y = train_test_split(train_data, train_label, test_size=0.2, random_state=1)
    # 训练分类器、进行预测、计算正确率
    clf = Clf_SVC(train_X, train_Y)
    pred = clf.predict(val_X)
    accuracy(val_Y, pred)
