# Deep learning test
Choisir un type de modèle parmi:  
`naive_bayes`  
`dnn`  
`cnn`  

Lancer l'entrainement et sauvegarder les prédictions, par exemple:
~~~bash
python train_and_predict.py --model_type naive_bayes
~~~

Les prédictions sont sauvegardées dans `Data/Test`.

Mes prédictions sont disponibles ici:  
`Data/Test/my_labels_naive_bayes.pkl`  
`Data/Test/my_labels_dnn.pkl`  
`Data/Test/my_labels_cnn.pkl`  


### Ma configuration
Python 3.6.1 :: Anaconda custom (64-bit)  
numpy==1.14.3  
scipy==0.19.1  
scikit-learn==0.20.0  
tensorflow==1.12.0  
Keras-Preprocessing==1.0.5  
