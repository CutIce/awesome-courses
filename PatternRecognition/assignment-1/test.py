# # d = {
# #     10: 20,
# #     2: 100,
# #     40: 1
# # }
# #
# # lst = [(11, 12), (20, 14), (90, 90), (1, 10), (2, 9), (3, 8), (4, 7), (5, 6)]
# # l = [2, 1, 20, 12, 233, 22]
# # print(l.index(max(l)))
# d = {(0,0):0, (1,1):0, (2,2):0}
# c = ((1,1), (1,3), (3, 1), (3, 3))
# is_ = (1,1) in c
# d1 = {i: 0 for i in c}
# print(is_)
# print(d1)
#
#
#
# # print(max(l))
# # # d_sorted = sorted(d.items(), key=lambda d:d[0], reverse=False)
# # # print(d_sorted)
# # #
# # # d_s1 = sorted(d.items(), key=lambda d:d[1], reverse=False)
# # # print(d_s1)
# # #
# # # print(d[2])
# #
# # lst_od0 = sorted(lst, key=lambda lst:lst[0], reverse=False)
# # # print(lst_od0)
# #
# # for i in range(len(lst_od0)):
# #     print(lst_od0[i][0])
# #
# # # lst_od1 = sorted(lst, key=lambda lst:lst[1],reverse=False)
# # # print(lst_od1)
# #
# #
# # res = [0] * 10
# # print(res)
#
# from sklearn.datasets import load_iris
# import csv
# #
# # iris = load_iris()
# #
# # print(type(iris))
# # print(len(iris))
# # print("keys: {}".format(iris.keys()))
# # print(iris.filename)
# # train_data = load_iris().data
# # train_label = load_iris().target
# # print(train_data)
# # print(train_label)
# # print(len(train_data))
# # print(len(train_label))

import codecs
from django.utils.encoding import smart_str
import chardet

path = 'E:/Documents/vdlrecords.1621264930.log'
new_path = 'E:/Documents/new_log.txt'
new_path2 = 'E:/Documents/new_log2.txt'
with open(path, 'rb') as f:
    f_type = chardet.detect(f.read())

print(f_type)
print(0)
with codecs.open(path, 'rb', encoding='ansi', errors='ignore') as f:
    content = smart_str(f.read())
    print(content)

print(1)
with codecs.open(new_path, 'wb', 'utf-8') as f:
    f.write(content)
print(2)

# content = codecs.open(path, 'r').read()
# content = content.decode(f_type['encoding'], 'ignore')
# codecs.open(new_path2, 'w', encoding='utf-8').write(content)

# print(lst)