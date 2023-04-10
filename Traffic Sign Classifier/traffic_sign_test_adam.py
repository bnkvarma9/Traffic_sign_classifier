import numpy as np
import os
import cv2
import csv
import random
from keras.utils import to_categorical
from keras.models import load_model
import matplotlib.pyplot as plt
model_adam=load_model("traffic_sign_model_100_adam")
model_sgd=load_model("traffic_sign_model_100_sgd")
categories=[]
categories=["0000"+str(i) for i in range(0,10)]
c2=["000"+str(i) for i in range(10,62)]
categories=categories+c2
datadir="BelgiumTSC_Testing/Testing"
test_data=[]

for category in categories:
    path=os.path.join(datadir,category)
    file=open(datadir+"/"+category+"/GT-"+category+".csv")
    csvreader=csv.reader(file,delimiter=';')
    header=next(csvreader)
    col=[header[0]]
    class_num=categories.index(category)
    
    for row in csvreader:
        img=cv2.imread(os.path.join(path,row[0]))
        siz=64

    
        new_array=cv2.resize(img,(siz,siz))
        test_data.append([new_array,class_num])

random.shuffle(test_data)
x_test=[]
y_test=[]

for feat,lab in test_data:
    x_test.append(feat)
    y_test.append(lab)
x_test=np.array(x_test).reshape(-1,64,64,3)
y_test=np.array(y_test)
x_test=x_test/255

y_test=to_categorical(y_test,62)

#output=model_adam.predict(x_test)
l=['uneven road','speed breaker','slippery road','Road ahead curves to the left side','Road ahead curves to the right side','Double curve ahead, to the left then to the right','Road bends right then left','Warning for children and minors',
   'Warning for bikes and cyclists','Cattle crossing','Roadworks ahead warning','Traffic light ahead',
   'Railroad crossing ahead with barriers','Cars not allowed - prohibited','road narrows ahead','Road gets narrow on the right side','Road gets narrow on the left side',
   'Crossroad ahead', 'side roads to right and left','Uncontrolled crossroad ahead','Give way to all traffic','single way traffic','Stop and give way to all traffic','No entry','cycle route ahead','load limit 5t 5','Truck','width limit 2m- speed 50','width limit 3m- speed 50'
   'prohibition','No left turn','No right turn','no overpassing','Speed limit','Mandatory shared path for pedestrians and cyclists',
 'Straight ahead only', 'Pass on right only, Turning right compulsory, Mandatory left','Driving straight ahead or turning right mandatory',
'Direction of traffic on roundabout','Cyclist must use mandatory path','Path for cyclists and pedestrians divided is compulsory','No parking','Stopping and parking forbidden',
 'No parking from the 1st to the 15th of the month.','No parking from the 16th to the 31st of the month.','Priority over oncoming traffic, road narrows',
 'Free and unrestricted parking for all types of vehicles.','Free and authorized parking only for people with reduced mobility.',
 'parking for motorcycles, cars and mini buses','Parking exclusively for lorries','Parking exclusively for buses','Parking mandatory on pavement or verge',
 'Begin of a residential area','End of the residential area','One-way traffic','Road ahead is a dead end','work prohibited','Crossing for pedestrians warning ahead',
 'Warning for bikes and cyclists','Indicating parking','Speed bump','End of priority road','Priority road']


model_adam.evaluate(x_test,y_test)
'''
y_pred=[]
for i in range(len(x_test)):
    x_t=x_test[i].reshape(1,64,64,3)
    y_pred.append(np.argmax(model_sgd.predict(x_t),axis=1))


y_t=[]
count=0
wrong=0
y_t=np.argmax(y_test,axis=1)
for i in range(len(x_test)):
    #print(y_pred[i]," ",y_t[i])
    if y_t[i]==y_pred[i]:
        count+=1
        #print(l[y_t[i]],"---- ",l[int(y_pred[i])])
    else:
        wrong+=1
print("correct: ",count," Wrong: ",wrong)

# adam-2433 correct, 101 wrong, total-2534
# sgd -2157 correct, 377 wrong
'''
'''
ele=x_test[0]
print(l[arr[0]])
plt.imshow(ele)
plt.draw()'''





        