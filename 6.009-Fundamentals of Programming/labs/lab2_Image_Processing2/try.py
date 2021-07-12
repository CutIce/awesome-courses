# # # x = 500
# # #
# # #
# # # def foo(y):
# # #     return x+y
# # #
# # # z = foo(307)
# # #
# # #
# # # print(x)
# # # print(foo)
# # # print(z)
# # #
# # #
# # # def bar(x):
# # #     x = 1000
# # #     return foo(308)
# # #
# # # w = bar('Hello')
# # #
# # # print()
# # # print(x)
# # # print(w)
# #
# #
# # x = 0
# #
# # def outer():
# #     x = 1
# #     def inner():
# #         print('in:', x)
# #
# #     inner()
# #     print('out:', x)
# #
# # print('glpbao', x)
# # outer()
# # # inner()
# # print('global', x)
#
#
# def add_n(n):
#     def inner(x):
#         return x+n
#
#     return inner
#
# add1 = add_n(1)
# add2 = add_n(2)
#
# print(add1(7))
# print(add2(3))
# print(add_n(8)(9))


functions = []
for i in range(5):
    def func(x):
        return x + i
    # print(i)
    functions.append(func)

for f in functions:
    print(f(0))