from flask import Flask, render_template, request
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow import keras
from tensorflow.keras.models import Sequential # 케라스의 Sequential()을 임포트
from tensorflow.keras.layers import Dense # 케라스의 Dense()를 임포트
from tensorflow.keras import optimizers # 케라스의 옵티마이저를 임포트
import numpy as np
import pandas as pd

app=Flask(__name__)

def kindDict(kind_, kind_dict): #사용자에게 입력받은 품종에 해당하는 품종코드 리턴
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

        #사용자에게 입력받기
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

        #원-핫 인코딩
        kindCd = np.array(kindCd_data, dtype=np.float64)
        neuter_list = np.array(['N', 'U', 'Y'])
        sex_list = np.array(['F', 'M', 'Q'])

        neuter_list = neuter_list.reshape(3)
        sex_list = sex_list.reshape(3)
        kindCd = kindCd.reshape(177)

        test_dict = {'kindNum': [kindDict(kindName, kind_dict)], 'neuterYn': [neuter], 'sexCd': [sex], 'weight': [weight],
                     'noticeDays': [notice],
                     'age2': [age]}

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

        #예측모델 불러오기
        model = keras.models.load_model('l2_model_Nadam')
        result = model.predict(test)
        adopt = result[0] * 100

        return render_template('index.html', adopt=np.round(adopt, 2))

if __name__ == '__main__':
    app.run(debug=True)
