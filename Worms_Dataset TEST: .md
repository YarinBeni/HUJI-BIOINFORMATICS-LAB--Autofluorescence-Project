# Yarin-s_Code
trying to give the autofluorescence problem a solution using computer vision and deep learning techniques.


Worms_Dataset TEST: 


# TEST the shape of the batch: 
# 1) change num into the wanted batch size
# 2) download database from git and update test_path accordingly
# 3) run test_batch_shape to get the batch shape and information ,
# batch[0] = DIC images, batch[1] = EFGP images(labels) , batch[2] = tuple contain path
# in the dataset folder:
# 5 samples, folder for each sample and in every sample 3 images: EGFP,DIC,MCHER.
# when DIC is image and EGFP is label ( not using MCHER)
# and max contain rectangle is (203, 933)
