# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 23:45:21 2019

@author: Areeb
"""

#importing libraries
import cv2
import numpy as np
import glob
import os
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import pickle


#list of folder names
letsgo = ['aunt_jemima_original_syrup','3m_high_tack_spray_adhesive',
          'campbells_chicken_noodle_soup','cheez_it_white_cheddar',
          'cholula_chipotle_hot_sauce','clif_crunch_chocolate_chip',
          'coca_cola_glass_bottle','detergent','expo_marker_red',
          'listerine_green','nice_honey_roasted_almonds','nutrigrain_apple_cinnamon',
          'palmolive_green','pringles_bbq','vo5_extra_body_volumizing_shampoo',
          'vo5_split_ends_anti_breakage_shampoo']

##########################################################################    
#########################-->1.Dataset Preprocessing<--####################
#############################(EXECUTE ONLY ONCE)##########################
##########################################################################

#croppig images
x=220
y=35
w=180
h=260

for name in letsgo:	
    os.mkdir(name)

for name in letsgo:
    path = 'train/'+name
    imagedict = [file for file in os.listdir(path) if os.path.splitext(file)[-1] == '.jpg']
    cropped=[]
    for file in glob.glob(path+'/*.jpg'):
        img = cv2.imread(file)
        cropped.append(img[y:y+h,x:x+w])
    for i in range(216):
        cv2.imwrite('cropped/'+name+'/'+imagedict[i],cropped[i])


##########################################################################    
#########################-->2.Creating Dictionary<--######################
############################(executed only once)##########################
##########################################################################


#creating dictionary of image names
dictionary=[]
for folder in letsgo:
    path = 'train/'+folder
    imagedict = [file for file in os.listdir(path) if os.path.splitext(file)[-1] == '.jpg']
    for name in imagedict:
        fullname =  folder+'_'+name
        dictionary.append(fullname)


##########################################################################    
###########################-->3.Creating X_train<--#######################
##################(saves X_train.txt to working directory)################
##########################################################################


#creating list of images
images=[]
for name in letsgo:
    for file in glob.glob('cropped/'+name+'/*.jpg'):
        images.append(cv2.imread(file))

xtrainlen = len(images)

#for surf
gray=[0]*xtrainlen
surf=[0]*xtrainlen
kpsurf=[0]*xtrainlen
dessurf=[0]*xtrainlen


#extracting surf features
for i in range(xtrainlen):
    surf[i] = cv2.xfeatures2d.SURF_create(extended=True)
    kpsurf[i], dessurf[i] = surf[i].detectAndCompute(gray[i],None)


#creating X_train by joining descriptors of all images
    
X_train = dessurf[0]   
for i in range(1,xtrainlen):
    X_train = np.append(X_train,dessurf[i],axis=0)
    

#saving X_train
np.savetxt('X_train.txt',X_train,fmt='%.6f')


##########################################################################    
###########################-->4.Creating X_test<--########################
##################(saves X_test.txt to working directory)#################
##########################################################################

test=[]
for file in glob.glob('sample_test/*.jpg'):
    test.append(cv2.imread(file))

xtestlen = len(test)

#for surf
graytest=[0]*xtestlen
surftest=[0]*xtrainlen
kpsurftest=[0]*xtrainlen
dessurftest=[0]*xtrainlen

#extracting surf features
for i in range(xtestlen):
    surftest[i] = cv2.xfeatures2d.SURF_create(extended=True)
    kpsurftest[i], dessurftest[i] = surftest[i].detectAndCompute(graytest[i],None)



#creating X_test by joining descriptors of all images

X_test = dessurftest[0]       
for i in range(1,xtestlen):
    X_test = np.append(X_test,dessurftest[i],axis=0)

#saving X_test
np.savetxt('X_test.txt',X_test,fmt='%.6f')

##########################################################################    
###################-->5.Loading X_train and X_test<--#####################
#########################(from working directory)#########################
##########################################################################

X_train = np.loadtxt('X_train.txt',dtype='float32')
X_test = np.loadtxt('X_test.txt',dtype='float32')


##########################################################################    
###################-->6.Fitting K-means to dataset<--#####################
############################(happens on cloud)############################
##########################################################################

clust = 500
kmeans = KMeans(n_clusters = clust, init = 'k-means++', random_state=0)
minikmeans = MiniBatchKMeans(n_clusters = clust, init = 'k-means++', random_state=0)
y_train = kmeans.fit_predict(X_train)
y_test = kmeans.predict(X_test)

##########################################################################    
#######-->7.Loading y_train and y_test from working directory<--##########
#####################(after downloading from drive)#######################
##########################################################################

y_train = np.loadtxt('y_train.txt',dtype='float32')
y_test = np.loadtxt('y_test.txt',dtype='float32')

    
##########################################################################    
###########################-->8.TF-IDF Score<--###########################
##########################################################################

#________________________________TRAINING________________________________#


imgclusters = []
lengths = []

for i in range(xtrainlen):
    lengths.append(len(dessurf[i]))

start = 0
for length in lengths:
    end = start+length
    imgclusters.append(y_train[start:end])
    start=end


frequencies = []
for features in imgclusters:
    features = features.tolist()
    img = []
    for i in range(clust):
        img.append(features.count(i))
    frequencies.append(img)

overall = []
ylist=y_train.tolist()
for i in range(clust):
    add=0
    for j in imgclusters:
        if i in j:
            add = add+1
    overall.append(add)

tfidf=[]
for i in range(xtrainlen):
    img = []
    for j in range(clust):
        ti = (frequencies[i][j]/lengths[i])*np.log10(xtrainlen/overall[j])
        img.append(ti)
    tfidf.append(img)
 

#________________________________TEST________________________________#

imgclusters_out = []
lengths_out = []

for i in range(xtestlen):
    lengths_out.append(len(dessurftest[i]))

start = 0
for length in lengths_out:
    end = start+length
    imgclusters_out.append(y_test[start:end])
    start=end

    
frequencies_out = []
for features in imgclusters_out:
    features = features.tolist()
    img = []
    for i in range(clust):
        img.append(features.count(i))
    frequencies_out.append(img)
    

tfidf_out=[]
for i in range(xtestlen):
    img = []
    for j in range(clust):
        ti = (frequencies_out[i][j]/lengths_out[i])*np.log10(xtrainlen/overall[j])
        img.append(ti)
    tfidf_out.append(img)
    
    
#############################################################################
####################-->9.Final Scores and Ranking<--#########################
#############################################################################

def normdot(p,q):
    dot = np.dot(p,q)
    dot = dot/(np.linalg.norm(p)*np.linalg.norm(q))
    return dot   

scores=[]
for i in range(xtestlen):
    score=[]
    for j in range(xtrainlen):
        score.append(normdot(tfidf[j],tfidf_out[i]))
    scores.append(score)
    
sortedscore=[]
for i in range(xtestlen):
    sortedscore.append(sorted(scores[i],reverse=True))

ranks=[]
for i in range(xtestlen):
    rank=[]
    for j in sortedscore[i]:
        rank.append(dictionary[scores[i].index(j)])
    ranks.append(rank)   
    
#############################################################################
#######################-->10.Saving output files<--##########################
#############################################################################
    
#storing pickle file
with open('ranks.pickle', 'wb') as f:
    pickle.dump(ranks, f)

testdict=[]
path = 'sample_test/'    
test = [file for file in os.listdir(path) if os.path.splitext(file)[-1] == '.jpg']
for name in test:
    if name.endswith('.jpg'):
        testdict.append(name[:-4])

destination='only surf/k='+str(clust)
os.mkdir(destination)
os.mkdir(destination+'/outputs')
for i in range(xtestlen):
    with open(destination+'/outputs/'+testdict[i]+'.txt', "w") as file:
        ranklist = ranks[i]
        for name in ranklist:
            file.write(name+'\n')