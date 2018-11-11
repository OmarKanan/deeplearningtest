# Deep learning test
Les chemins de fichiers sont configurables dans `config.py`  
Mes prédictions sont disponibles ici:  
`Data/Test/my_labels_naive_bayes.pkl`  
`Data/Test/my_labels_dnn.pkl`  
Les 2 modèles donnent des scores similaires.

### Modèle Naive Bayes
Pour lancer l'entrainement et enregistrer les prédictions:
~~~bash
python train_and_predict.py --model_type naive_bayes
~~~
Un fichier sera généré à `Data/Test/labels_naive_bayes.pkl`

### Modèle Deep Neural Network
Pour lancer l'entrainement et enregistrer les prédictions:
~~~bash
python train_and_predict.py --model_type dnn
~~~
Attention c'est long ! (environ 20 minutes sur ma machine)

Un fichier sera généré à `Data/Test/labels_dnn.pkl`

### Ma configuration
Python 3.6.1 :: Anaconda custom (64-bit)  
numpy==1.14.3  
scipy==0.19.1  
scikit-learn==0.20.0  
tensorflow==1.12.0  
