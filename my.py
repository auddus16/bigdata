from flask import Flask, render_template, request
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import pandas as pd

kind=pd.read_csv("testDogKind.csv")
kindCd=pd.read_csv("kindCd.csv")

kind=kind.dropna(axis=0)
kindCd_data = kindCd.dropna(axis=0)

print(kind)
print(kindCd)

#품종과 품종코드를 분리하여 array에 저장
kind1=np.array(kind.KNm)
kind2=np.array(kind.kindCd, dtype=np.float64)

kind_dict={}
for i in range(len(kind2)):
    kind_dict[kind1[i]]=kind2[i]
print(kind_dict)

X = tf.placeholder(dtype = tf.float64, shape = [None, 186])
Y = tf.placeholder(dtype = tf.float64, shape = [None, 1])  # 가격(실제값)이 들어있는 placeholder

a = tf.Variable(tf.random_uniform([186, 1], dtype = tf.float64))
b = tf.Variable(tf.random_uniform([1], dtype = tf.float64))  # y절편

y = tf.sigmoid(tf.matmul(X, a) + b)

# 배추 가격을 결정하는 4가지 변화 요인을 입력받습니다.
kind = input('품종 :')
neuter = input('중성화 여부 :')
sex = input('성별 :')
weight = float(input('몸무게 : '))
notice= float(input('공고일 수: '))
age= float(input('나이: '))

for cd, num in kind_dict.items():
    if(kind==cd):
        kind=num
        break
print(kind)

kindCd = np.array(kindCd_data, dtype = np.float64)
neuter_list=np.array(['N', 'U', 'Y'])
sex_list=np.array(['F', 'M', 'Q'])

neuter_list=neuter_list.reshape(3)
sex_list=sex_list.reshape(3)
kindCd = kindCd.reshape(177)

print(neuter_list)
print(sex_list)
print(kindCd)
print(kindCd.shape)

test={'kindNum':[kind], 'neuterYn':[neuter], 'sexCd':[sex], 'weight':[weight], 'noticeDays':[notice], 'age2':[age]}
test=pd.DataFrame(test)
print(test)

# kindNum을 원핫 인코딩
kindCd= pd.concat((pd.get_dummies(test.kindNum, columns=kindCd), pd.DataFrame(columns=kindCd))).fillna(0)
test.drop(['kindNum'], axis='columns', inplace=True)
test=pd.concat([test, kindCd], axis=1)

neuter_list = pd.concat((pd.get_dummies(test.neuterYn, columns=neuter_list), pd.DataFrame(columns=neuter_list))).fillna(0)
test.drop(['neuterYn'], axis='columns', inplace=True)
test=pd.concat([test, neuter_list], axis=1)

sex_list = pd.concat((pd.get_dummies(test.sexCd, columns=sex_list), pd.DataFrame(columns=sex_list))).fillna(0)
test.drop(['sexCd'], axis='columns', inplace=True)
test=pd.concat([test, sex_list], axis=1)

print(test)

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, './dog.ckpt')
    test=np.array(test, dtype=np.float64)
#     print(sess.run(y, feed_dict = {X: test}))
    result = sess.run(y, feed_dict = {X: test})
    print("result:{}".format(result[0]))