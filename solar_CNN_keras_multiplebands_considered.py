# coding: utf-8
""" Prediction with CNN
input: fisheye image
out: Generated Power
クロスバリデーションもする
とりあえずkeras
"""

# library
import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import os
import sys
import time
import seaborn as sns
import glob
import Load_data_for_considered as ld
import argparse
# matplotlib.use('Agg')

from keras.models import Sequential, model_from_json, model_from_yaml
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, Adadelta, RMSprop
# from sklearn.metrics import mean_absolute_error, mean_squared_error

SAVE_dir = "./RESULT/CNN_keras_multiple_bands_considered100/"
if not os.path.isdir(SAVE_dir):
    os.makedirs(SAVE_dir)

def CNN_model1(activation="relu", loss="mean_squared_error", optimizer="Adadelta"):
    """
    INPUT -> [CONV -> RELU -> CONV -> RELU -> POOL]*2 -> [FC -> RELU]*2 -> OUT
    """
    model = Sequential()

    model.add(ZeroPadding2D((1, 1), input_shape=(layer, height, width)))
    model.add(Convolution2D(16, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(16, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model


def CNN_model2(activation="relu", loss="mean_squared_error", optimizer="Adadelta"):
    """
    INPUT -> [CONV -> RELU -> POOL]*2 -> FC -> RELU -> OUT
    """
    model = Sequential()

    model.add(ZeroPadding2D((1, 1), input_shape=(layer, height, width)))
    model.add(Convolution2D(16, 3, 3, border_mode='same'))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024, activation=activation))
    model.add(Dropout(0.5))

    model.add(Dense(1))

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model


def CNN_model3(
    activation="relu",
    loss="mean_squared_error",
    optimizer="Adadelta",
    layer=0,
    height=0,
    width=0):
    """
    INPUT -> [CONV -> RELU] -> OUT
    """
    model = Sequential()

    model.add(ZeroPadding2D((1, 1), input_shape=(layer, height, width)))
    model.add(Convolution2D(16, 3, 3, border_mode='same'))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(1))

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model


def data_plot(model, target, img, batch_size=10, date="hoge", save_csv=True):

    num = []
    time = []
    for i in range(target[:, 0].shape[0]):
        if i % 50 == 0:
            num.append(i)
            time.append(target[:, 0][i])
        if i == target[:, 0].shape[0] - 1:
            num.append(i)
            time.append(target[:, 0][i])
    img_ = img.transpose(0, 3, 1, 2).copy()
    pred = model.predict(img_, batch_size=batch_size, verbose=1).copy()
    # if pred.shape:
    #     print(pred.shape)
    # if type(pred):
    #     print(type(pred))
    # print(target[1].shape)
    plt.figure()
    plt.plot(pred, label="Predicted")
    plt.plot(target[:, 1], label="Observed")
    plt.legend(loc='best')
    plt.title("Prediction tested on"+date)
    plt.xlabel("Time")
    plt.ylabel("Generated Power[kW]")
    plt.ylim(0, 25)
    plt.xticks(num, time)

    pred_ = pred.reshape(pred.shape[0])
    if save_csv is True:
        save_target_and_prediction(target=target[:, 1], pred=pred_, title=date)

    filename = date + "_data"
    i = 0
    while os.path.exists(SAVE_dir+'{}{:d}.png'.format(filename, i)):
        i += 1
    plt.savefig(SAVE_dir+'{}{:d}.png'.format(filename, i))


def save_target_and_prediction(target, pred, title):

    pred_df = pd.DataFrame(pred)
    target_df = pd.DataFrame(target)
    df = pd.concat([target_df, pred_df], axis=1)
    df.columns = ["TARGET", "PREDICTION"]
    df.to_csv(SAVE_dir+"Error_csv_"+title+".csv")


def main():

    """ 画像の日付リストの獲得
    img_20170101 = np.array
    みたいな感じで代入していく
    また、ディレクトリのパスをdictionalyに入れておくことで、targetのロードのときに役たてる
    """
    img_dir_path_dic = {}
    # img_name_list = []
    date_list = []
    error_date_list = []
    img_tr = []
    target_tr = []

    img_month_list = os.listdir(DATA_DIR)
    img_month_list.sort()

    for month_dir in img_month_list:
        if not month_dir.startswith("."):
            im_dir = os.path.join(DATA_DIR, month_dir)
            img_day_list = os.listdir(im_dir)
            img_day_list.sort()
            for day_dir in img_day_list:
                if not day_dir.startswith("."):
                    dir_path = os.path.join(im_dir, day_dir)
                    img_dir_path_dic[day_dir[:8]] = dir_path
                    # img_name_list.append("img_"+day_dir[:8])

    """ ターゲットの読み込み
    target_20170101 = np.array
    みたいな感じで代入していく
    dictionalyに保存したpathをうまく利用
    """

    target_month_list = os.listdir(TARGET_DIR)
    target_month_list.sort()

    # テストの時用
    # COUNT = 0  # COUNTで読み込む日数を指定する

    for month_dir in target_month_list:
        # if not month_dir.startswith("."):
        if month_dir == "201705":
            im_dir = os.path.join(TARGET_DIR, month_dir)
            target_day_list = os.listdir(im_dir)
            target_day_list.sort()
            for day_dir in target_day_list:
                if not day_dir.startswith("."):
                    file_path = os.path.join(im_dir, day_dir)
                    # if COUNT < 2:
                    #     COUNT += 1
                    print("---- TRY ----- " + day_dir[3:11])
                    try:
                        target_tmp = ld.load_target(csv=file_path, imgdir=img_dir_path_dic[day_dir[3:11]])
                        img_tmp = ld.load_image(imgdir=img_dir_path_dic[day_dir[3:11]], size=(100, 100), norm=True)
                        if len(target_tmp) == len(img_tmp):
                            target_tr.append(target_tmp)  # (number, channel)
                            date_list.append(day_dir[3:11])
                            img_tr.append(img_tmp)
                            print("   OKAY")
                        else:
                            print("   数が一致しません on "+day_dir[3:11])
                            print("   target: {}".format(len(target_tmp)))
                            print("   img: {}".format(len(img_tmp)))
                            error_date_list.append(day_dir[3:11])

                    except:
                        print("   Imageデータがありません on "+day_dir[3:11])


    # errorの日を保存
    with open(SAVE_dir+"error_date.txt", "w") as f:
        f.write(str(error_date_list))
    """
    ['20170112', '20170114', '20170118', '20170420', '20170428', '20170604', '20170615']
    でエラーがでる
    """

    print("Data Load Done. Starting traning.....")
    print("training on days " + str(date_list))
    test_error_list = []

    # traning
    for i in range(len(date_list)):

        print("-----Training on "+str(date_list[i])+"-----")
        training_start_time = time.time()

        ts_img = 0
        ts_target = 0
        ts_img_pool = 0
        ts_target_pool = 0
        # i=1
        ts_img_pool = img_tr.pop(i)
        ts_target_pool = target_tr.pop(i)
        ts_img = ts_img_pool.copy()
        ts_target = ts_target_pool.copy()

        img_tr_all = 0
        target_tr_all = 0
        img_tr_all = np.concatenate((
            img_tr[:]
        ), axis=0)
        target_tr_all = np.concatenate((
            target_tr[:]
        ), axis=0)

        # テストデータと訓練データから、訓練データでとった平均を引く
        mean_img = ld.compute_mean(image_array=img_tr_all)
        img_tr_all -= mean_img
        ts_img -= mean_img

        """
        # 多バンド化の実行します
        # 1.トレーニング画像
        # 2.テスト画像
        # 3.トレーニングターゲット
        # 4.テストターゲット
        """
        # 1.トレーニング画像
        # リストの定義
        tmp = []  # 一時的に画像をプールする
        BAND_NUM = 3  # まとめる枚数の定義 (共通)
        img_n_sec = []  # n枚バンド化した画像を追加するリスト
        img_band_array = 0  # 最終的に使うnumpy配列
        for j in range(len(img_tr_all)):
            if j <= len(img_tr_all)-BAND_NUM:
                tmp_nparray = np.concatenate((
                    img_tr_all[j:j+BAND_NUM]
                ), axis=2)  # 画像が(224, 224, 3)の形で入っているので、axisは2
                img_n_sec.append(tmp_nparray)  # バンド化した画像をリストにappend
                tmp = []  # プールリストの初期化
            else:
                break
        img_band_array = np.array(img_n_sec, dtype=float)
        print(img_band_array.shape)

        # 2.テスト画像
        # リストの定義
        tmp = []  # 一時的に画像をプールする
        img_n_sec_ts = []  # n枚バンド化した画像を追加するリスト
        img_band_array_ts = 0  # 最終的に使うnumpy配列
        for j in range(len(ts_img)):
            if j <= len(ts_img)-BAND_NUM:
                tmp_nparray = np.concatenate((
                    ts_img[j:j+BAND_NUM]
                ), axis=2)
                img_n_sec_ts.append(tmp_nparray)  # バンド化した画像をリストにappend
                tmp = []  # プールリストの初期化
            else:
                break
        img_band_array_ts = np.array(img_n_sec_ts, dtype=float)
        print(img_band_array_ts.shape)

        # 次にターゲット(発電量)
        # 3.トレーニングターゲット
        # リストの定義
        tmp = []  # 一時的に画像をプールする
        target_n_sec = []  # n枚バンド化した発電量を追加するリスト
        target_band_array = 0  # 最終的に使うnumpy配列
        for j in range(len(target_tr_all)):
            if j <= len(target_tr_all)-BAND_NUM:
                tmp = np.float32(target_tr_all[j+BAND_NUM-1, 1])
                target_n_sec.append(tmp)  # バンド化したターゲットをリストにappend
                tmp = []  # プールリストの初期化
            else:
                break
        target_band_array = np.array(target_n_sec, dtype=float)  # ここでエラーでた
        target_band_array = np.concatenate((
            target_tr_all[:-(BAND_NUM-1), 0][:, np.newaxis],
            target_band_array[:, np.newaxis],
            target_tr_all[:-(BAND_NUM-1), 2][:, np.newaxis]
        ), axis=1)  # 配列を(data数, 3)に直さないと、データプロットのところでエラー出る
        print(target_band_array.shape)

        # 4.テストターゲット
        # リストの定義
        tmp = []  # 一時的に画像をプールする
        target_n_sec_ts = []  # n枚バンド化した発電量を追加するリスト
        target_band_array_ts = 0  # 最終的に使うnumpy配列
        for j in range(len(ts_target)):
            if j <= len(ts_target)-BAND_NUM:
                tmp = np.mean(np.float32(ts_target[j+BAND_NUM-1, 1]))
                target_n_sec_ts.append(tmp)  # バンド化したターゲットをリストにappend
                tmp = []  # プールリストの初期化
            else:
                break
        target_band_array_ts = np.array(target_n_sec_ts, dtype=float)
        target_band_array_ts = np.concatenate((
            ts_target[:-(BAND_NUM-1), 0][:, np.newaxis],
            target_band_array_ts[:, np.newaxis],
            ts_target[:-(BAND_NUM-1), 2][:, np.newaxis]
        ), axis=1)
        print(target_band_array_ts.shape)

        print("Bandalized DONE")
        """
        バンド化おわり
        img_band_array = 0  # 最終的に使うnumpy配列
        target_band_array = 0  # 最終的に使うnumpy配列
        """


        # transpose for CNN INPUT shit
        img_band_array = img_band_array.transpose(0, 3, 1, 2)
        print(img_band_array.shape)
        # set image size
        layer = img_band_array.shape[1]
        height = img_band_array.shape[2]
        width = img_band_array.shape[3]

        print("Image and Target Ready")

        # parameter
        activation = ["relu", "sigmoid"]
        optimizer = ["adam", "adadelta", "rmsprop"]
        nb_epoch = [10, 25, 50]
        batch_size = [5, 10, 15]

        # model set
        model = None
        model = CNN_model3(
            activation="relu",
            optimizer="Adadelta",
            layer=layer,
            height=height,
            width=width)
        # plot_model(model, to_file='CNN_model.png')

        # initialize check
        data_plot(
            model=model, target=target_band_array_ts, img=img_band_array_ts, batch_size=10,
            date=date_list[i], save_csv=True)

        early_stopping = EarlyStopping(patience=3, verbose=1)

        # Learning model
        hist = model.fit(img_band_array, target_band_array[:, 1],
                         epochs=nb_epoch[0],
                         batch_size=batch_size[1],
                         validation_split=0.1,
                         callbacks=[early_stopping])
        data_plot(
            model=model, target=target_band_array_ts, img=img_band_array_ts, batch_size=10,
            date=date_list[i], save_csv=True)
        # evaluate
        try:
            img_tmp = img_band_array_ts.transpose(0, 3, 1, 2)
            score = model.evaluate(img_tmp, target_band_array_ts[:, 1], verbose=1)
            print("Evaluation "+date_list[i])
            print('TEST LOSS: ', score[0])
            test_error_list.append(score[0])
        except:
            print("error in evaluation")

        try:
            model.save(SAVE_dir+"model_{}".format(str(date_list[i]))+".h5")
        except:
            print("error in save model")

        # put back data
        img_tr.insert(i, ts_img_pool)
        target_tr.insert(i, ts_target_pool)
        # img_tr.insert(1, img_band_array_ts)
        # target_tr.insert(1, target_band_array_ts)

        tr_elapsed_time = time.time() - training_start_time
        print("elapsed_time:{0}".format(tr_elapsed_time)+" [sec]")

    # error_lossの日を保存
    with open(SAVE_dir+"test_loss.txt", "w") as f:
        f.write(str(test_error_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="../data/PV_IMAGE/",
        help="choose your data (image) directory"
    )
    parser.add_argument(
        "--target_dir",
        default="../data/PV_CSV/",
        help="choose your target dir"
    )
    args = parser.parse_args()
    DATA_DIR, TARGET_DIR = args.data_dir, args.target_dir

    # 時間の表示
    start = time.time()
    main()
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time)+" [sec]")







# DATA_DIR = "../data/PV_IMAGE/"
# TARGET_DIR = "../data/PV_CSV/"


















# end
