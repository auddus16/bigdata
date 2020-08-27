from flask import Flask, render_template, request
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import pandas as pd

app=Flask(__name__)

X = tf.placeholder(dtype = tf.float64, shape = [None, 186])
Y = tf.placeholder(dtype = tf.float64, shape = [None, 1])  # 가격(실제값)이 들어있는 placeholder

a = tf.Variable(tf.random_uniform([186, 1], dtype = tf.float64))
b = tf.Variable(tf.random_uniform([1], dtype = tf.float64))  # y절편

y = tf.sigmoid(tf.matmul(X, a) + b)

saver=tf.train.Saver()

sess=tf.Session()
sess.run(tf.global_variables_initializer())

saver.restore(sess, './model/dog.ckpt')


def kindDict(kind_, kind_dict):
    for cd, num in kind_dict.items():
        if (kind_ == cd):
            kind_ = num
            return kind_


@app.route("/")
@app.route("/index")
def index():
    if request.method == 'GET':
        return render_template('index.html')

@app.route('/', methods=['POST'])
def make_prediction():
    if request.method == 'POST':

        kindName = request.form['kindName']
        neuter = request.form['neuter']
        sex = request.form['sex']
        weight = float(request.form['weight'])
        notice = float(request.form['notice'])
        age = float(request.form['age'])

        kind = pd.read_csv("testDogKind.csv")
        kindCd = pd.read_csv("kindCd.csv")

        kind = kind.dropna(axis=0)
        kindCd_data = kindCd.dropna(axis=0)

        # 품종과 품종코드를 분리하여 array에 저장
        kind1 = np.array(kind.KNm)
        kind2 = np.array(kind.kindCd, dtype=np.float64)

        kind_dict = {}
        for i in range(len(kind2)):
            kind_dict[kind1[i]] = kind2[i]

        kindCd = np.array(kindCd_data, dtype=np.float64)
        neuter_list = np.array(['N', 'U', 'Y'])
        sex_list = np.array(['F', 'M', 'Q'])

        neuter_list = neuter_list.reshape(3)
        sex_list = sex_list.reshape(3)
        kindCd = kindCd.reshape(177)

        # def kindDict(kind_):
        #     for cd, num in kind_dict.items():
        #         if (kind_ == cd):
        #             kind_ = num
        #             return kind_

        test_dict = {'kindNum': [kindDict(kindName, kind_dict)], 'neuterYn': [neuter], 'sexCd': [sex], 'weight': [weight],
                     'noticeDays': [notice],
                     'age2': [age]}
        # test=pd.DataFrame(data=np.array([[kindDict(kindName)],[neuter],[sex],[weight],[notice],[age]]), columns=['kindNum', 'neuter','sexCd','weight','noticeDays', 'age'])
        test = pd.DataFrame(test_dict)

        kindCd = pd.concat((pd.get_dummies(test.kindNum, columns=kindCd), pd.DataFrame(columns=kindCd))).fillna(0)
        test.drop(['kindNum'], axis='columns', inplace=True)
        test = pd.concat([test, kindCd], axis=1)

        neuter_list = pd.concat(
            (pd.get_dummies(test.neuterYn, columns=neuter_list), pd.DataFrame(columns=neuter_list))).fillna(0)
        test.drop(['neuterYn'], axis='columns', inplace=True)
        test = pd.concat([test, neuter_list], axis=1)

        sex_list = pd.concat((pd.get_dummies(test.sexCd, columns=sex_list), pd.DataFrame(columns=sex_list))).fillna(0)
        test.drop(['sexCd'], axis='columns', inplace=True)
        test = pd.concat([test, sex_list], axis=1)

        new_data = np.array(test, dtype=np.float64)

        result = sess.run(y, feed_dict={X: new_data})

        adopt = result[0] * 100

        return render_template('index.html', adopt=adopt)

if __name__ == '__main__':
    app.run(debug=True)