<p align="center"><img width=12.5% src="https://github.com/eduardmagerusan/medicaldiagnosisai/blob/fd2727a6bc66a47773a5962dcaf8f9b78b7680a2/media/40489153.jpg"></p>
<p align="center"><img width=60% src="https://github.com/eduardmagerusan/medicaldiagnosisai/blob/eaea30925e5f314aad4e474a48a9d35caf627b19/media/medicaldiagnosisai.png"></p>


# Basic Overview
In this Notebook I use a pretrained densenet121 Model for predicting multiple category diagnosis from chest x-ray images. The location of 
abnormalities are visulized using GradCam algorithm. 

# Dataset
The project uses the ChestX-ray8 dataset which contains 108,948 frontal-view X-ray images of 32,717 unique patients.
- Each image in the data set contains multiple text-mined labels identifying 14 different pathological conditions.
- These in turn can be used by physicians to diagnose 8 different diseases.
- The data will be used to develop a single model that will provide binary classification predictions for each of the 14 labeled pathologies.
- We will predict 'positive' or 'negative' for each of the abnormalities.

# Results
![Bildschirmfoto 2022-06-15 um 22 41 45](https://user-images.githubusercontent.com/84686184/173923155-12b3c43c-17b4-432c-b697-9cec70a2d110.png)

The model correctly predicts the mass and pneumothorax. We can also notice that the model picks up the absence of edema and cardiomegaly. The model 
correctly highlights the area of interest. This visualization is useful for error analysis. We can notice if the model is indeed looking at the expected 
area for making the prediction. 

#WebApp
![Bildschirmfoto 2022-07-28 um 18 13 58](https://user-images.githubusercontent.com/84686184/181587547-c9cd23ba-e93b-41a6-9417-9c017a696ba6.png)

# References
- https://www.kaggle.com/code/redwankarimsony/nih-chest-x-ray8-classifier-cnn-visualization/notebook
- https://github.com/LaurentVeyssier/Chest-X-Ray-Medical-Diagnosis-with-Deep-Learning
