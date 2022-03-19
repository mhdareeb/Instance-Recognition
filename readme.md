1. zip file contanins 4 python files - sift.py, surf.py, sift_surf.py and yolo_opencv.py
   zip file contains ranks.pickle file generated on final ranklist using pickle
   zip file contains md5hash.txt which has the md5 hash of ranks.pickle
   zip file contains outputs folder which has the ranklists for each image in sample_test
   zip file contains a project report named CS783_Assignment1.pdf

2. sift.py - Extracts only sift features from training dataset and predicts y_train accordingly.
3. surf.py - Extracts only surf features from training dataset and predicts y_train accordingly
4. sift&surf.py - Extracts both sift and surf features from training dataset and predicts y_train using both
5. yolo_opencv.py - contains YOLO implementation in opencv which we have kept for improving our model in future

------------------------------------------------------------------------------------------------------------

A) Basic operation for all 3 codes -->
   -------------------------------

1.Dataset Preprocessing --> crops all images in training set with left=220, width=180, top=35, height=260
(to be executed only once at the start of program)

2.Creating Dictionary --> creates dictionary of image names for future reference

3.Creating X_train --> extracts features from training dataset and creates corresponding X_train
				   --> extracts only sift features for sift.py
				   --> extracts only surf features for surf.py
				   --> extracts both sift and surf features for sift&surf.py
				       and creates X_train by joining normalized sift and surf features
(also saves X_train to working directory)

4.Creating X_test --> extracts features from testset and creates corresponding X_test
				  --> extracts only sift features for sift.py
				  --> extracts only surf features for surf.py
				  --> extracts both sift and surf features for sift&surf.py
				      and creates X_test by joining normalized sift and surf features
(also saves X_test to working directory)

5.Loading X_train and X_test --> loads X_train and X_test from current working directory
(which were stored there in step 4)

6.Fitting K-Means to dataset --> fits K-Means clustering model to training set and creates y_train and y_test
(this part is done on cloud and files y_train.txt and y_test.txt are downloaded to working directory)

7.Loading y_train and y_test --> loads y_train and y_test from current working directory
(which were downloaded there in step 6)

8.TF-IDF score --> calculates TF-IDF score of training and test datasets

9.Final scores and ranking --> calculates cosine similarity score using TF-IDF scores of step 8 and 
				gives final rankings

10.Saving output files --> saves output files neatly


B) Output Files --> contains rank files for given sample test data for k=10000
   ------------

------------------------------------------------------------------------------------------------------------------
