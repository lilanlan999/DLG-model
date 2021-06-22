import scipy.io
import os
import numpy as np
from matplotlib import pyplot as plt
from keras import backend as K
from keras.models import load_model
from keras.models import Model
from resnet import ResnetBuilder
import xlwt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import auc
from keras import optimizers
from scipy import interp


def random_shuffle(data,label):
    randnum = np.random.randint(0, 1234)
    np.random.seed(randnum)
    np.random.shuffle(data)
    np.random.seed(randnum)
    np.random.shuffle(label)
    return data,label
def process_train(path1,path2):
    x_paths = []
    y_labels = []
    image_paths = os.listdir(path1)
    for path_img in image_paths:
        x_paths.append(path1 + "/" + path_img)
        y_labels.append(1)
    image_paths = os.listdir(path2)
    for path_img in image_paths:
        x_paths.append(path2 + "/" + path_img)
        y_labels.append(0)
    random_shuffle(x_paths, y_labels)
    batch_res = []
    y_res = []
    for i in range(len(x_paths)):
        batch_res.append(read_single(x_paths[i]))
        y_res.append(y_labels[i])
    #y = (np.arange(2) == y_res[:, None]).astype(int)
    return np.array(batch_res),np.array(y_res)

def process_test(path1,path2):
    x_paths = []
    y_labels = []
    image_paths = os.listdir(path1)
    for path_img in image_paths:
        x_paths.append(path1 + "/" + path_img)
        y_labels.append(1)
    image_paths = os.listdir(path2)
    for path_img in image_paths:
        x_paths.append(path2 + "/" + path_img)
        y_labels.append(0)
    random_shuffle(x_paths, y_labels)
    batch_res = []
    y_res = []
    for i in range(len(x_paths)):
        batch_res.append(read_single(x_paths[i]))
        y_res.append(y_labels[i])
    y_res = np.array(y_res)
    #y = (np.arange(2) == y_res[:, None]).astype(int)
    return np.array(batch_res),np.array(y_res)

def read_single(img_path):
    img = scipy.io.loadmat(img_path).get('Amp')
    img_full = np.array(img)
    img_full = np.expand_dims(img_full, axis=0)
    img_full = np.concatenate((img_full, img_full, img_full), axis=0)
    return img_full

def ptint_out_in_sheet(dense1_output):
    for obj_index in range(len(dense1_output)):
        for index, item in enumerate(dense1_output[obj_index]):
            sheet.write(index, obj_index, str(item))
    f.save("./adhc_out.xls")

def model_compile(model):
    # DIM_ORDERING = {'th', 'tf'}
    # for ordering in DIM_ORDERING:
    # K.set_image_dim_ordering("tf")
    K.set_image_data_format('channels_first')
    model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer=optimizers.Adam(lr=1e-3))    # assert True, "Failed to compile with '{}' dim ordering".format(ordering)

def calculate_metric(gt, pred):
    gt2 = []
    pred2 = []
    for i in range(len(pred)):
        pred2.append(0 if pred[i,0]>pred[i,1] else 1)
        gt2.append(0 if gt[i,0]>gt[i,1] else 1)
    confusion = confusion_matrix(gt2,pred2)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    print('Accuracy:',(TP+TN)/float(TP+TN+FP+FN))
    print('Sensitivity:',TP / float(TP+FN))
    print('Specificity:',TN / float(TN+FP))
    return (TP+TN)/float(TP+TN+FP+FN),TP / float(TP+FN),TN / float(TN+FP)

if __name__=='__main__':
    n=1
    K.set_image_data_format('channels_first')
    k_total = 10  # 交叉验证次数
    f = xlwt.Workbook()
    sheet = f.add_sheet('tprs_adhc', cell_overwrite_ok=True)
    accu = []
    sen = []
    spe = []
    tprs = []
    aucs = []
    fpr_s = []
    tpr_s = []
    pdf = []
    X_train, Y_train = process_train("data/ad", "data/hc")
    for k_count in range(k_total):
       model = ResnetBuilder.build_resnet_18((3, 776, 776), 2)
       model_compile(model)
       mean_fpr = np.linspace(0, 1, 100)
       skf = StratifiedKFold(n_splits=5)
       for i, (train, test) in enumerate(skf.split(X_train, Y_train)):
          print('--------------------- {}fold - --------------------'.format(i))
          print('len-train:', len(train))
          print('len-test:', len(test))
          print(" ")
          y = (np.arange(2) == (Y_train[train])[:, None]).astype(int)
          ytest = (np.arange(2) == (Y_train[test])[:, None]).astype(int)
          history = model.fit(X_train[train],y,batch_size=2,validation_data = (X_train[test],ytest),epochs=1,shuffle=True)
          X_test, Y_test = process_test("data/test/ad", "data/test/hc")
          Yy_test = (np.arange(2) == Y_test[:, None]).astype(int)
          Y_pred = model.predict(X_test)
          accuracy, sensitivity, specificity = calculate_metric(Yy_test, Y_pred)
          print("**************           accuracy: {}".format(accuracy))
          print("**************           sensitivity: {}".format(sensitivity))
          print("**************           specificity: {}".format(specificity))
          Y_pred = [np.argmax(y) for y in Y_pred]
          fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_pred)
          tprs.append(interp(mean_fpr, fpr, tpr))
          tprs[-1][0] = 0.0
          roc_auc = metrics.auc(fpr, tpr)
          aucs.append(roc_auc)
          result = [accuracy, sensitivity, specificity]
          accu.append(result[0])
          sen.append(result[1])
          spe.append(result[2])
          n = n+1
          fpr_s.append(fpr)
          tpr_s.append(tpr)
          mean_tpr = np.mean(tprs, axis=0)
          mean_tpr[-1] = 1.0
          pdf = [fpr_s, tpr_s, tprs, mean_fpr, mean_tpr, aucs, accu, sen, spe]
          ptint_out_in_sheet(pdf)
          del model
          model = ResnetBuilder.build_resnet_18((3, 776, 776), 2)
          model_compile(model)
