import numpy as np


def manhattan_distance(x1, x2):
    return np.sum(np.absolute(x1 - x2), axis=-1)


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2, axis=-1))


class ModelKNN:
    
    def __init__(self, k=3, dis=euclidean_distance):
        """
        具体的 KNN 模型

        :param k: KNN 中 K 的值
        :param dis: 用于衡量数据之间的距离标准
        """
        self.k = k
        self.get_distance = dis
        self.train_data = None
        self.train_label = None
    
    def fit(self, train_data, train_label):
        """
        KNN 本身没有显式的学习过程，只需要把数据存下来

        :param train_data: shape=(N,D) 的 numpy.ndarray, N 为训练数据数量, D 为数据维度
        :param train_label: shape=(N,) 的 numpy.ndarray, N 为训练数据数量
        :return:
        """
        assert train_data.shape[0] == train_label.shape[0]
        assert train_data.shape[1]
        
        self.train_data = train_data
        self.train_label = train_label
    
    def predict(self, test_data):
        """
        用存下来的数据进行预测的过程，需要你来补全代码

        :param test_data: shape=(M,D) 的 numpy.ndarray, M 为 测试数据数量, D 为数据维度
        :return: shape=(M,) 的 numpy.ndarray, M 为 测试数据数量
        """
        if len(test_data) == 0:
            return []
        predict_labels = [0] * test_data.shape[0]
        
        ##############################
        #    write your code here    #
        ##############################

        return np.array(predict_labels)


class AlgorithmKNN:
    
    def __init__(self, fold_num=5):
        """

        :param fold_num:  N-fold 交叉验证的中的 N 值
        """
        self.MAX_K = 7
        self.fold_num = fold_num
        self.model = None
    
    def fit(self, train_data, train_label):
        """
        寻找合适 KNN 模型的算法

        :param train_data: shape=(N,D) 的 numpy.ndarray, N 为训练数据数量, D 为数据维度
        :param train_label: shape=(N,) 的 numpy.ndarray, N 为训练数据数量
        :return:
        """
        assert train_data.shape[0] == train_label.shape[0]
        assert train_data.shape[1]
        
        # 随机打乱训练数据
        idx = np.arange(train_data.shape[0])
        np.random.shuffle(idx)
        train_data = train_data[idx]
        train_label = train_label[idx]
        
        # 尝试不同 K，用 N-fold 交叉验证的方式来进行选择
        best_acc = -1
        
        for k in range(1, min(self.MAX_K, train_data.shape[0] // self.fold_num + 1)):
            for distance_method in [manhattan_distance, euclidean_distance]:
                model = ModelKNN(k, distance_method)
                acc = self.cross_validation(model, self.fold_num, train_data, train_label)
                print("K={}, {}, acc={}".format(model.k, distance_method.__name__, round(acc, 2)))
                if acc > best_acc:
                    best_acc = acc
                    self.model = model
        
        print("[choose K={}, distance_method={}]".format(self.model.k, self.model.get_distance.__name__))
    
    def predict(self, test_data):
        """
        用选择出最好的 KNN 模型来进行预测

        :param test_data: shape=(M,D) 的 numpy.ndarray, M 为 测试数据数量, D 为数据维度
        :return: shape=(M,) 的 numpy.ndarray, M 为 测试数据数量
        """
        return self.model.predict(test_data)
    
    @staticmethod
    def cross_validation(model, fold_num, train_data, train_label):
        """
        N-fold 交叉验证

        :param model: ModelKNN 对象，待测试的模型
        :param fold_num: N-fold 交叉验证的中的 N 值
        :param train_data: shape=(N,D) 的 numpy.ndarray, N 为训练数据数量, D 为数据维度
        :param train_label: shape=(N,) 的 numpy.ndarray, N 为训练数据数量
        :return: N-fold 交叉验证的平均准确率
        """
        L = train_data.shape[0] // fold_num
        split_data = [train_data[L * i: L * i + L] for i in range(fold_num - 1)] + [train_data[L * (fold_num - 1):]]
        split_label = [train_label[L * i: L * i + L] for i in range(fold_num - 1)] + [train_label[L * (fold_num - 1):]]
        
        acc = []
        for i in range(fold_num):
            # split_data 和 split_label 是两个包含 fold_num 个 numpy.ndarray 的数组
            # train 的时候将 (fold_num-1) 个 numpy.ndarray 的数组 concatenate 起来，valid ate 的时候用剩下的 1 个
            val_data = split_data[i]
            val_label = split_label[i]
            train_data = np.concatenate(split_data[:i] + split_data[i + 1:])
            train_label = np.concatenate(split_label[:i] + split_label[i + 1:])
            if train_data.shape[0] == 0:
                continue
            
            model.fit(train_data, train_label)
            if val_data.shape[0] == 0:
                acc.append(0)
            else:
                acc.append(np.mean(model.predict(val_data) == val_label))
        return sum(acc) / len(acc)


def test_case_demo():
    """
    四维数据样例 D=4, N=5, M=2

    :return: train_data(N,D), train_label(N,), test_data(M,D), test_label(M,)
    """
    train_data = np.array([
        [1, 2, 3, 4],
        [4, 2, 3, 1],
        [12, 12, 13, 14],
        [14, 12, 13, 11],
        [12, 14, 15, 16]
    ])
    train_label = np.array([0, 0, 1, 1, 1])
    
    test_data = np.array([
        [3, 4, 4, 2],
        [18, 14, 15, 16]
    ])
    test_label = np.array([0, 1])
    return train_data, train_label, test_data, test_label


def test_case():
    """
    二维数据样例 D=2, N=100, M=2, 训练数据来自两个高斯分布的混合

    :return: train_data(N,D), train_label(N,), test_data(M,D), test_label(M,)
    """
    
    mean = (1, 2)
    cov = np.array([[73, 0], [0, 22]])
    x = np.random.multivariate_normal(mean, cov, (80,))
    
    mean = (16, -5)
    cov = np.array([[21.2, 0], [0, 32.1]])
    y = np.random.multivariate_normal(mean, cov, (20,))
    
    train_data = np.concatenate([x, y])
    train_label = np.concatenate([
        np.zeros((80,), dtype=int),
        np.ones((20,), dtype=int)
    ])
    
    test_data = np.array([
        [3, 4],
        [18, -2]
    ])
    test_label = np.array([0, 1])
    
    return train_data, train_label, test_data, test_label


def test():
    train_data, train_label, test_data, test_label = test_case()
    algo = AlgorithmKNN()
    algo.fit(train_data, train_label)
    print(np.mean(algo.predict(test_data) == test_label))


if __name__ == "__main__":
    test()
