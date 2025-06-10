import matplotlib
matplotlib.use('Agg')
from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
import os
from django.core.files.storage import FileSystemStorage
import cv2
import io
import base64
import numpy as np
import matplotlib.pyplot as plt
from FCM import FCM
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from VanillaConGan import getVanillaCoGan, generateImages
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras.applications import VGG16
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, AveragePooling2D
from keras.layers import Convolution2D
from keras.models import Sequential, load_model, Model
import pickle
import pyswarms as ps
from SwarmPackagePy import testFunctions as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

global uname, gan_X, gan_Y, labels, rf_cls, pso, vgg_size

path = "GanDataset"
labels = []
gan_X = []
gan_Y = []
accuracy = []
precision = []
recall = []
fscore = []

for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if name not in labels:
            labels.append(name.strip())

def getLabel(name):
    index = -1
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index

def calculateMetrics(algorithm, predict, y_test):
    global accuracy, precision, recall, fscore
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    conf_matrix = confusion_matrix(y_test, predict)
    random_probs = [0 for i in range(len(y_test))]
    p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)
    ns_fpr, ns_tpr, _ = roc_curve(y_test, predict, pos_label=1)
    return conf_matrix, p_fpr, p_tpr, ns_fpr, ns_tpr

classifier = RandomForestClassifier()

gan_size = len(os.listdir('GanDataset/benign'))
if gan_size < 2:
    malign = np.load('model/malignant_X.npy')
    benign = np.load("model/benign_X.npy")
    malign_gan_model, benign_gan_model = getVanillaCoGan()
    for i in range(0, 100):
        img = malign[i]
        generateImages(malign_gan_model, img, "malignant", i)
    for i in range(0, 100):
        img = benign[i]
        generateImages(benign_gan_model, img, "benign", i)    
if os.path.exists('model/gan_X.npy'):
    gan_X = np.load('model/gan_X.npy')
    gan_Y = np.load('model/gan_Y.npy')
else:
    for root, dirs, directory in os.walk(path):
        for j in range(len(directory)):        
            name = os.path.basename(root)
            if 'Thumbs.db' not in directory[j]:
                img = cv2.imread(root+"/"+directory[j])
                img = cv2.resize(img, (32, 32))
                gan_X.append(img)
                label = getLabel(name)
                gan_Y.append(label)
                print(name+" "+str(label))                
    gan_X = np.asarray(gan_X)
    gan_Y = np.asarray(gan_Y)
    np.save('model/gan_X',gan_X)
    np.save('model/gan_Y',gan_Y)

print(gan_Y)
print(gan_Y.shape)
print(np.unique(gan_Y, return_counts=True))

indices = np.arange(gan_X.shape[0])
np.random.shuffle(indices)
gan_X = gan_X[indices]
gan_Y = gan_Y[indices]
gan_Y = to_categorical(gan_Y)

X_train, X_test, y_train, y_test = train_test_split(gan_X, gan_Y, test_size=0.2) #split dataset into train and test

vgg_model = VGG16(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), include_top=False, weights='imagenet')
for layer in vgg_model.layers:
    layer.trainable = False
vgg_model = Sequential()
vgg_model.add(Convolution2D(32, (3 , 3), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
vgg_model.add(MaxPooling2D(pool_size = (2, 2)))
vgg_model.add(Convolution2D(32, (3, 3), activation = 'relu'))
vgg_model.add(MaxPooling2D(pool_size = (2, 2)))
vgg_model.add(Flatten())
vgg_model.add(Dense(units = 64, activation = 'relu'))
vgg_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
vgg_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
if os.path.exists("model/vgg_weights.keras") == False:
    model_check_point = ModelCheckpoint(filepath='model/vgg_weights.keras', verbose = 1, save_best_only = True)
    hist = vgg_model.fit(X_train, y_train, batch_size = 32, epochs = 30, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    f = open('model/vgg_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    vgg_model.load_weights("model/vgg_weights.keras")

features_extractor = Model(vgg_model.inputs, vgg_model.layers[-2].output)#create VGG  model
vgg_features = features_extractor.predict(gan_X)  #extracting VGG features
vgg_size = vgg_features.shape[1]
gan_Y = np.argmax(gan_Y, axis=1)
#PSO function
def f_per_particle(m, alpha):
    global vgg_features, gan_Y, classifier
    total_features = vgg_features.shape[1]
    if np.count_nonzero(m) == 0:
        X_subset = vgg_features
    else:
        X_subset = vgg_features[:,m==1]
    classifier.fit(X_subset, gan_Y)
    P = (classifier.predict(X_subset) == gan_Y).mean()
    j = (alpha * (1.0 - P) + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))
    return j

def f(x, alpha=0.88):
    n_particles = x.shape[0]
    j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
    return np.array(j)

def runPSO():
    global vgg_features
    if os.path.exists("model/pso.npy"):
        pso = np.load("model/pso.npy")
    else:
        options = {'c1': 0.5, 'c2': 0.5, 'w':0.9, 'k': 5, 'p':2}
        dimensions = vgg_features.shape[1] # dimensions should be the number of features
        optimizer = ps.discrete.BinaryPSO(n_particles=10, dimensions=dimensions, options=options) #CREATING PSO OBJECTS 
        cost, pso = optimizer.optimize(f, iters=35)#OPTIMIZING FEATURES
        np.save("model/pso", pso)
    return pso  # PSO WILL SELECT IMPORTANT FEATURES WHERE VALUE IS 1

pso = runPSO()
vgg_features = vgg_features[:,pso==1]  # PSO WILL SELECT IMPORTANT FEATURES WHERE VALUE IS 1
X_train, X_test, y_train, y_test = train_test_split(vgg_features, gan_Y, test_size=0.2) #split dataset into train and test
f = open('model/data.pckl', 'rb')
data = pickle.load(f)
f.close()
X_train, X_test, y_train, y_test = data

svm_cls = svm.SVC(C=2)
svm_cls.fit(X_train, y_train)
predict = svm_cls.predict(X_test)
acc = accuracy_score(y_test, predict)
conf_matrix, p_fpr, p_tpr, ns_fpr, ns_tpr = calculateMetrics("SVM", predict, y_test)

lr_cls = LogisticRegression()
lr_cls.fit(X_train, y_train)
predict = lr_cls.predict(X_test)
conf_matrix, p_fpr, p_tpr, ns_fpr, ns_tpr = calculateMetrics("Logistic Regression", predict, y_test)

rf_cls = RandomForestClassifier(n_estimators=100)
rf_cls.fit(X_train, y_train)
predict = rf_cls.predict(X_test)
conf_matrix, p_fpr, p_tpr, ns_fpr, ns_tpr = calculateMetrics("Random Forest", predict, y_test)
    

def UploadDataset(request):
    if request.method == 'GET':
        global gan_X, gan_Y, labels
        benign = np.load('model/benign_X.npy')
        malign = np.load('model/malignant_X.npy')
        output = "Dataset Images Loading Completed<br/>Total Images Found in CBIS-DDSM Dataset = "+str(benign.shape[0] + malign.shape[0])+"<br/>"
        output += "Class Labels found in Dataset = "+str(labels)
        context= {'data': output+"<br/><br/><br/><br/>"}
        return render(request, 'AdminScreen.html', context)

def RunGan(request):
    if request.method == 'GET':
        global gan_X, gan_Y
        output = "Vanilla & Conditional GAN Synthetic Images Generation Completed<br/>"
        output += "Total Images available in dataset after applying Vanilla Conditional GAN = "+str(gan_X.shape[0])+"<br/>"
        context= {'data': output+"<br/><br/><br/><br/>"}
        return render(request, 'AdminScreen.html', context)

def RunPso(request):
    if request.method == 'GET':
        global gan_X, gan_Y, pso, vgg_features, X_train, vgg_size
        output = "VGG16 Features Extraction & PSO Features Selection Process Completed<br/>"
        output += "Total features available in each image = "+str(gan_X.shape[1] * gan_X.shape[2] * gan_X.shape[3])+"<br/>"
        output += "Number of features extracted using VGG16 from Original Input Image features : "+str(vgg_size)+"<br/>"
        output += "Number of features selected using PSO = "+str(X_train.shape[1])
        context= {'data': output+"<br/><br/><br/><br/>"}
        return render(request, 'AdminScreen.html', context)

def TrainML(request):
    if request.method == 'GET':
        global conf_matrix, p_fpr, p_tpr, ns_fpr, ns_tpr
        global accuracy, precision, recall, fscore, labels
        cols = ['Algorithm Name', 'Accuracy', 'Precision', 'Recall', 'Fscore']
        output = '<table border="1" align="center" width="100%"><tr>'
        font = '<font size="" color="black">'
        for i in range(len(cols)):
            output += "<td>"+font+cols[i]+"</font></td>"
        algorithms = ['SVM', 'Logistic Regression', 'Random Forest']
        output += "</tr>"
        for i in range(len(algorithms)):
            output += "<tr><td>"+font+algorithms[i]+"</font></td>"
            output += "<td>"+font+str(accuracy[i])+"</font></td>"
            output += "<td>"+font+str(precision[i])+"</font></td>"
            output += "<td>"+font+str(recall[i])+"</font></td>"
            output += "<td>"+font+str(fscore[i])+"</font></td></tr>"
        output += "</table><br/>"
        fig, axs = plt.subplots(1,2,figsize=(12, 5))
        ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g", ax=axs[0]);
        ax.set_ylim([0,len(labels)])
        axs[0].set_title("Random Forest Confusion matrix")        
        axs[1].plot(p_fpr, p_tpr, linestyle='--', color='orange',label="True classes")
        axs[1].plot(ns_fpr, ns_tpr, linestyle='--', label='Predicted Classes')
        axs[1].set_title("Random Forest ROC AUC Curve")
        axs[1].set_xlabel('False Positive Rate')
        axs[1].set_ylabel('True Positive rate')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.clf()
        plt.cla()
        context= {'data':output, 'img': img_b64}
        return render(request, 'AdminScreen.html', context) 

def getSegmented(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    cluster = FCM(img, image_bit=8, n_clusters=4, m=2, epsilon=0.5, max_iter=100)
    cluster.form_clusters()
    result=cluster.result
    plt.imshow(result)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    if os.path.exists("CancerApp/static/test.png"):
        os.remove("CancerApp/static/test.png")
    plt.savefig("CancerApp/static/test.png",bbox_inches='tight')
    plt.close()
    segment = cv2.imread("CancerApp/static/test.png")
    segment = cv2.resize(segment, (600, 400))
    segment = cv2.cvtColor(segment, cv2.COLOR_BGR2RGB)
    return segment

def PredictAction(request):
    if request.method == 'POST':
        global uname, labels, pso, rf_cls
        filename = request.FILES['t1'].name
        image = request.FILES['t1'].read() #reading uploaded file from user
        if os.path.exists("CancerApp/static/"+filename):
            os.remove("CancerApp/static/"+filename)
        with open("CancerApp/static/"+filename, "wb") as file:
            file.write(image)
        file.close()
        test_vgg_model = load_model("model/vgg_weights.keras")
        test_img = cv2.imread("CancerApp/static/"+filename)
        test_img = cv2.resize(test_img, (32, 32))
        test_img = test_img.reshape(1,32,32,3)
        test_features_extractor = Model(test_vgg_model.inputs, test_vgg_model.layers[-2].output)#create VGG  model
        test_vgg_features = test_features_extractor.predict(test_img)  #extracting VGG features
        print(test_vgg_features.shape)
        test_vgg_features = test_vgg_features[:,pso==1]
        print(test_vgg_features.shape)
        predict = rf_cls.predict(test_vgg_features)
        predict = predict[0]
        predict = labels[predict]
        segmented = getSegmented("CancerApp/static/"+filename)
        img = cv2.imread("CancerApp/static/"+filename)
        img = cv2.resize(img, (600, 400))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.putText(img, predict, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
        figure, axis = plt.subplots(nrows=1, ncols=2,figsize=(8, 5))
        axis[0].set_title("Original Image")
        axis[1].set_title("FCM Segmented Image")
        axis[0].imshow(img)
        axis[1].imshow(segmented)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()    
        plt.clf()
        plt.cla()
        context= {'data':'Predicted As '+predict, 'img': img_b64}
        return render(request, 'AdminScreen.html', context)   

def Predict(request):
    if request.method == 'GET':
        return render(request, 'Predict.html', {}) 

def AdminLogin(request):
    if request.method == 'GET':
       return render(request, 'AdminLogin.html', {})  

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def AdminLoginAction(request):
    if request.method == 'POST':
        global uname
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        page = "AdminLogin.html"
        status = "Invalid Login" 
        if "admin" == username and "admin" == password:
            page = "AdminScreen.html"
            status = "Welcome Admin"
        context= {'data': status}
        return render(request, page, context)



