import numpy as np


def manhattan_distance(x1, x2):
    return np.sum(np.absolute(x1 - x2), axis=-1)


def eucliden_distance(x1, x2):
    return np.sqrt(np.sum(np.square(x1-x2)))


class KNNModel:

    def __init__(self, k, dis_func=manhattan_distance):
        self.k = k
        self.dis_func = dis_func
        self.train_data = None
        self.train_label = None

    def fit(self, train_data, train_label):
        assert train_data.shape[1]
        assert train_data.shape[0] == train_label.shape[0]

        self.train_data = train_data
        self.train_label = train_label

    def predict(self, test_data, test_label):
        assert test_data.shape[0] == test_label.shape[0]

        if len(test_data) == 0:
            return []

        predict_label = [0] * test_data.shape[0]

        for i in range(len(test_data)):
            lst = []
            for j in range(len(self.train_data)):
                dis = self.dis_func(test_data[i], self.train_data[j])
                lst.append( (j, dis) )
            lst_odr = sorted(lst, key=lambda lst:lst[1], reverse=False)
            res = [0] * max(20, self.k)
            for num in self.k:
                kind = self.train_label[lst_odr[num][0]]
                res[kind] = res[kind] + 1
            predict_label[i] = res.index(max(res))

        return np.array(predict_label)


class AlgorithmKNN:

    def __init__(self, max_k=3, fold_num=5):
        self.MAX_K = max_k
        self.fold_num = fold_num
        self.model = None

    def fit(self, train_data, train_label):
        assert train_data.shape[0] == train_label.shape[0]
        assert train_data.shape[1]

        idx = np.arrange(train_data.shape[0])
        np.random.shuffle(idx)

        train_data = train_data[idx]
        train_label = train_label[idx]

        best_acc = -1

        for k in range(1, min(self.MAX_K, train_data.shape[0] // self.fold_num)):
            for dist_mothod in [manhattan_distance, eucliden_distance]:
                tmp_model = KNNModel(k=k, dis_func=dist_mothod)
                acc = self.cross_validation(tmp_model, self.fold_num, train_data, train_label)
                print("K = {}, {}, acc={}".format(tmp_model.k, dist_mothod.__name__, round(acc, 2)))
                if acc > best_acc:
                    best_acc = acc
                    self.model = tmp_model

        print("Choose K = {}, distance method = {}".format(self.model.k, self.model.dis_func.__name__))

    def predict(self, test_data):
        return self.model.predict(test_data)

    @staticmethod
    def cross_validation(self, fold_num, train_data, train_label):
        l = train_data.shape[0] // fold_num
        split_data = [train_data[l * i : l* i + l] for i in range(fold_num - 1)] + [train_data[l * fold_num - 1: ]]
        