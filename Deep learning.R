
#----------------------- Configuration du post de travail----------------------------
install.packages('devtools',  type="win.binary")
install.packages('reticulate', type="win.binary")
install.packages("tensorflow",type="win.binary")
install.packages("keras",type="win.binary")
## Autre alternative
require('devtools')
install_github("rstudio/reticulate",force =TRUE)
install_github("rstudio/tensorflow", force = TRUE)
install_github("rstudio/keras",force = TRUE)

library('reticulate')
library('tensorflow')
library('keras')
install_tensorflow()
install_keras()
library('dplyr') # Pour l'utilisation des pipes
library('zeallot') # Pour l'utilisation de l'affectation multiple
library('Hmisc')

#------------------------------------Model de deep learning----------------------------------------
## définir un modèle: en utilisant la fonction keras_model_sequential()
##(uniquement pour les piles linéaires de couches, qui est de loin l'architecture de réseau la plus courante)
library(keras)
model = keras_model_sequential() %>% 
  layer_dense(units = 32, input_shape = c(784)) %>% 
  layer_dense(units = 10, activation = "softmax")

# Définition du modèle à l'aide de l'API fonctionnelle
input_tensor = layer_input(shape = c(784))      
output_tensor = input_tensor %>%
  layer_dense(units = 32, activation = "relu") %>% 
  layer_dense(units = 10, activation = "softmax") 
model2 = keras_model(inputs = input_tensor, outputs = output_tensor)

# Etape de compilation avec une seule fonction de perte
model %>% compile(optimizer = optimizer_rmsprop(lr = 0.0001),
                  loss = "mse",metrics = c("accuracy"))
# Processus d'apprentissage
model %>% fit(input_tensor, target_tensor, batch_size = 128, epochs = 10)

#---------------------- CLASSIFICATION DES CRITIQUES DE CINÉMA : UNE CLASSIFICATION BINAIRE------------------------------
## Le jeu de données IMDB
### Chargement du jeu de données
library(keras)
dataset_imdb(num_words = 10000)
imdb = dataset_imdb(num_words = 10000)
c(c(train_data, train_labels), c(test_data, test_labels)) %<-%imdb  

### Pour le plaisir voici comment décoder cette liste de mots
library(keras)
dataset_imdb_word_index()
word_index = dataset_imdb_word_index()
reverse_word_index = names(word_index)
names(reverse_word_index) = word_index
decoded_review = sapply(train_data[[1]], function(index) {
    word = if (index >= 3) reverse_word_index[[as.character(index - 3)]]
    if (!is.null(word)) word else "?"
  })
##word_index est une liste nommée mappant des mots à un index entier.
##reverse_word_index, mappant les indices entiers aux mots
## Notez que les indices sont décalés de 3, car 0, 1 et 2 sont des indices réservés pour « padding », « start of sequence » et « unknown ».

# Préparation des données
## Transformation des listes en tenseur à fin d'alimenter le réseau de neurone.
vectorize_sequences = function(sequences, dimension = 10000) {
  results = matrix(0, nrow = length(sequences), ncol = dimension)
  for (i in 1:length(sequences))
    results[i, sequences[[i]]] = 1
  return(results)
}

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

## Convertion des étiquettes d'entier en numérique.
y_train = as.numeric(train_labels)
y_test = as.numeric(test_labels)

## Maintenant, les données sont prêtes à être introduites dans un réseau de neurones.

# Construiction de notre réseau
# Nous construisons notre premier réseau constitué de :
##-Deux couches intermédiaires avec 16 unités cachées chacune et untilisant relu comme fonction d'activation.
##-Une troisième couche avec une unité, utilisant sigmoïde comme fonction d'activation et qui produira la prédiction scalaire concernant le sentiment de l'examen en cours.
library(keras)
model <- keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

##Configuration du modèle avec l'optimiseur rmsprop et la fonction de perte binary_crossentropy. 
##Notons qu'on surveillera également la précision pendant l'entraînement :
model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("accuracy"))

## Validation du modèle
val_indices = 1:10000
x_val = x_train[val_indices,]
partial_x_train = x_train[-val_indices,]
y_val = y_train[val_indices]
partial_y_train = y_train[-val_indices]

## Formation de notre modèle
history = model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)

## L'objet history contient une méthode plot() qui vous permet de visualiser la formation et métriques de validation par epochs :
plot(history)

## Entraînons un nouveau réseau à partir de zéro pendant quatre epochs, puis évaluons-le sur les données de test.
# 1. Construction du réseau
library(keras)
model = keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")
# 2. Configuration du réseau
model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)
# 3. Formation du réseau
model %>% fit(x_train, y_train, epochs = 4, batch_size = 512)
# 4. Evaluation du réseau su l'échantillon test
results = model %>% evaluate(x_test, y_test)
# Cette approche assez naïve atteint une précision de 88%. Avec des approches de pointe, nous devrions pouvoir atteindre près de 95%

# Utiliser un réseau formé pour générer des prédictions sur de nouvelles données
model %>% predict(x_test[1:10,])


#------------------CLASSIFICATION DES FILS DE PRESSE : UNE CLASSIFICATION MULTICLASSE----------------
#### Le jeu de données Reuters.
# chargement de la base
library(keras)
reuters <-dataset_reuters(num_words = 10000)
c(c(train_data, train_labels), c(test_data, test_labels)) %<-% reuters
train_data
# comment vous pouvez le décoder en mots, au cas où vous seriez curieux
index_mots <- ensemble_données_Reuters_mot_index()
reverse_word_index <- noms(word_index)
decoded_newswire <- sapply(train_data[[1]], function(index) {
  mot <- if (index >= 3) reverse_word_index[[as.character(index>=3)]]
  if (!is.null(word)) word else "?"
})

# Préparation des données : 
##Nous pouvons vectoriser les données avec exactement le même code que dans l'exemple précédent.

vectorize_sequences <- function(
  sequences, dimension = 10000) {
  results = matrix(0, nrow = length(sequences), ncol = dimension)
  for (i in 1:length(sequences))
    results[i, sequences[[i]]] = 1
  return(results)
}
x_train <- vectorize_sequences(train_data) 
x_test <- vectorize_sequences(test_data)

## vectorisation des étiquettes(encodage onhot)
one_hot_train_labels <- to_categorical(train_labels)
one_hot_test_labels <- to_categorical(test_labels)

# Construction du réseau de neurone
model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu",
              input_shape = c(10000)) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 46, activation = "softmax")

# Compilation du modèle
model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

# Formation du modèle :
## Echantillon de validation
val_indices = 1:1000
x_val = x_train[val_indices,]
partial_x_train = x_train[- val_indices,]
y_val = one_hot_train_labels[val_indices,]
partial_y_train = one_hot_train_labels[- val_indices,]

## Maintenant, formons le réseau pendant 20 époques
history <- model%>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)

## Et enfin, nous pouvons afficher ses courbes de perte et de précision
plot(history)

## Entraînons un nouveau réseau à partir de zéro pendant neuf époques, puis évaluons-le sur l'ensemble de test.
### construction du modèle
model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu", input_shape = c(10000)) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 46, activation = "softmax")

### Compilation du modèle
model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

### formation du modèle
history <- model%>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 9,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)

### Evaluation du modèle sur l'échantillon test
results <- model %>% evaluate(x_test, one_hot_test_labels)
results


#--------------------------Prévoir les prix des logements : un exemple de régression-----------------------
# Chargement de l'ensemble de données sur le logement de Boston
library(keras)
dataset <- dataset_boston_housing()
c(c(train_data, train_targets), c(test_data, test_targets)) %<-% dataset
describe(data.frame(train_targets))
describe(data.frame(test_targets))

# Préparation des données (Normalisation des données)
mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
train_data <- scale(train_data, center = mean, scale = std)
test_data <-scale(test_data, center = mean, scale = std)

# Construction du réseau
build_model <- function(){
  model <- keras_model_sequential() %>%
    layer_dense(units = 64, activation = "relu",
                input_shape = dim(train_data)[[2]]) %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 1)
  
  model %>% compile(
    optimizer = "rmsprop",
    loss = "mse",
    metrics = c("mae"))
}

# Validation du modèle à l'aide de la validation K-fold
## Découper les données disponibles pour l'entrainement en k partitions 
k <- 4
indices <- sample(1:nrow(train_data))
folds <- cut(indices, breaks = k, labels = FALSE)
num_epochs <- 100 
all_scores <- c()
for (i in 1:k) {
  ## Regrouper k-1 partitions pour l'entrainnement et évaluer le modèle sur la partition i.
  cat("\n\n processing fold #", i, "\n---------------\n")
  val_indices <- which(folds == i, arr.ind = TRUE)
  val_data <- train_data[val_indices,]
  val_targets <- train_targets[val_indices]
  partial_train_data <- train_data[-val_indices,]
  partial_train_targets <- train_targets[-val_indices]
  ## Construition du modèle Keras (déjà compilé)
  model <- build_model()
  ## Formation du modèle sur les k-1 partitions
  model %>% fit(partial_train_data, partial_train_targets, 
                epochs = num_epochs, batch_size = 1, verbose = 0)
  ## Evaluation du modèle sur la partition i
  results <- model %>% evaluate(val_data, val_targets, verbose = 0)
  print(results)
  all_scores <- c(all_scores, results[2])
}
## Erreur moyenne quadratique
cat('moyenne mae :',mean(all_scores))

# Entraînement du réseau un peu plus longtemps : 500 époques, en conservant une trace de ses performances àchque époques
num_epochs <- 500
all_mae_histories <- NULL

for (i in 1:k) {
  # Regrouper k-1 partitions pour l'entrainnement et évaluer le modèle sur la partition i.
  cat("\n\n processing fold #", i, "\n---------------\n")
  val_indices <- which(folds == i, arr.ind = TRUE)
  val_data <- train_data[val_indices,]
  val_targets <- train_targets[val_indices]
  partial_train_data <- train_data[-val_indices,]
  partial_train_targets <- train_targets[-val_indices]
  # Construition du modèle Keras (déjà compilé)
  model <- build_model()
  # Formation du modèle sur les k-1 partitions en conservant les traces ses performances à chaque époques
  history <- model%>% fit(
    partial_train_data, partial_train_targets,
    validation_data = list(val_data, val_targets),
    epochs = num_epochs, batch_size = 1, verbose = 0
  )
  print(history)
  # Score de validation
  mae_history <- history$metrics$val_mae
  all_mae_histories <- rbind(all_mae_histories, mae_history)
}

# Visualisation des performances du modèle
average_mae_history <- data.frame(
  epoch = seq(1:ncol(all_mae_histories)),
  validation_mae = apply(all_mae_histories, 2, mean)
)
ggplot(average_mae_history, aes(x = epoch, y = validation_mae)) + geom_line()

# Formation finale du modèle
model <- build_model()
model %>% fit(train_data, train_targets, 
              epochs = 52, batch_size = 4, verbose = 0)

result <-model%>% evaluate(test_data, test_targets)
result

#--------------------------------Réseaux de neurones convolutionnels---------------------------------------

##----------- Préparation des données
## dossier des données d'entrainement initial
base_train_dir <- "training_set"
dir.create(base_train_dir)
## dossier des données de test initial
base_test_dir <- "test_set"
dir.create(base_test_dir)
## repertoir d'entrainement (contenant les données à utiliser pour l'entrainement avant la validation du modèle)
train_dir <-file.path("train")
dir.create(train_dir)
## dossier de validation
validation_dir <-file.path("validation")
dir.create(validation_dir)
## dossier de test
test_dir <-file.path("test")
dir.create(test_dir)
## sous dossier d'entrainement des données de chats
train_cats_dir <-file.path(train_dir, "cats")
dir.create(train_cats_dir)
##sous dossier d'entrainement des données de chiens
train_dogs_dir <-file.path(train_dir, "dogs")
dir.create(train_dogs_dir)
## sous dossier de validation des données de chats
validation_cats_dir <-file.path(validation_dir, "cats")
dir.create(validation_cats_dir)
##  sous dossier de validation des données de chiens
validation_dogs_dir <-file.path(validation_dir, "dogs")
dir.create(validation_dogs_dir)
##  sous dossier de test des données de chats
test_cats_dir <-file.path(test_dir, "cats")
dir.create(test_cats_dir)
## sous dossier de test des données de chiens
test_dogs_dir <-file.path(test_dir, "dogs")
dir.create(test_dogs_dir)

### Préparation des données des chiens
## Données d'entrainement
fnames <-paste0("training_set/dogs/dog.", 1:3000, ".jpg")
file.copy(file.path(base_train_dir, fnames),
          file.path(train_dogs_dir))

## Données de validation
fnames <-paste0("training_set/dogs/dog.", 3001:4000, ".jpg")
file.copy(file.path(base_train_dir, fnames),
          file.path(validation_dogs_dir))
 
## Données de test
fnames <-paste0("test_set/dogs/dog.", 4001:5000, ".jpg")
file.copy(file.path(base_test_dir, fnames),
          file.path(test_dogs_dir))

### Préparation des données des chats
## Données d'entrainement
fnames <-paste0("training_set/cats/cat.", 1:3000, ".jpg")
file.copy(file.path(base_train_dir, fnames),
          file.path(train_cats_dir))

## Données de validation
fnames <-paste0("training_set/cats/cat.", 3001:4000, ".jpg")
file.copy(file.path(base_train_dir, fnames),
          file.path(validation_cats_dir))

## Données de test
fnames <-paste0("test_set/cats/cat.", 4001:5000, ".jpg")
file.copy(file.path(base_test_dir, fnames),
          file.path(test_cats_dir))

#------------Construction du réseau
library(keras)
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), 
                activation = "relu", input_shape = c(150, 150, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3),
                activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3),
                activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), 
                activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

#----------- Compilation du modèle
model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-4),
  metrics = c("acc")
)

#-------- Prétraitement des données
# Redimensionne toutes les images au 1/255
train_datagen <- image_data_generator(rescale = 1.0/255) 
validation_datagen <- image_data_generator(rescale = 1.0/255) 
# 
train_generator <-flow_images_from_directory(
  train_dir, # Répertoire cible
  train_datagen, # Générateur de données d'entraînement
  target_size = c(150, 150), # Redimensionne toutes les images à 150 × 150
  batch_size = 20, 
  class_mode = "binary" # Comme vous utilisez la perte binary_crossentropy, vous avez besoin d'étiquettes binaires.
)

#
validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)

# ----- Formation du modèle
history <- model %>% fit_generator(
   train_generator,
   steps_per_epoch = 100,
   epochs = 60,
   validation_data = validation_generator,
   validation_steps = 60
)
#------ Courbe d'entrainement
plot(history)

#-----Enrégistrement du modèle
model%>% save_model_hdf5("ISEP2_cats_Vs_dogs")

# ---- Formation finale du modèle

## Ajouter les échantillons de valitation à l'ensemble d'entrainement
fnames <-paste0("dogs/dog.", 3001:4000, ".jpg")
file.copy(file.path(validation_dir, fnames),
          file.path(train_dogs_dir))
fnames <-paste0("cats/cat.", 3001:4000, ".jpg")
file.copy(file.path(validation_dir, fnames),
          file.path(train_cats_dir))

## Prétraitement des données
# Redimensionne toutes les images au 1/255
train_datagen <- image_data_generator(rescale = 1.0/255) 
test_datagen <- image_data_generator(rescale = 1.0/255)
# 
train_generator <-flow_images_from_directory(
  train_dir, # Répertoire cible
  train_datagen, # Générateur de données d'entraînement
  target_size = c(150, 150), # Redimensionne toutes les images à 150 × 150
  batch_size = 20, 
  class_mode = "binary" # Comme vous utilisez la perte binary_crossentropy, vous avez besoin d'étiquettes binaires.
)
#
test_generator <-flow_images_from_directory(
  test_dir, # Répertoire cible
  test_datagen, # Générateur de données d'entraînement
  target_size = c(150, 150), # Redimensionne toutes les images à 150 × 150
  batch_size = 20, 
  class_mode = "binary" # Comme vous utilisez la perte binary_crossentropy, vous avez besoin d'étiquettes binaires.
)

# Entrainement du modèle pré-formé
history <- model %>% fit_generator(
        train_generator,
        steps_per_epoch = 100,
        epochs = 42 )

# Evaluation du modèle
model %>% evaluate_generator(test_generator, steps = 2)

#-----Enrégistrement du modèle
model%>% save_model_hdf5("ISEP2_cats_Vs_dogs")

#----Utiliser le modèle formé pour classer des chiens des chats
dt = 'yatoute'
pre_trait = function(dt) {
  dt_datagen <- image_data_generator(rescale = 1.0/255) 
  dt_generator <-flow_images_from_directory(
  dt, # Répertoire cible
  dt_datagen, # Générateur de données d'entraînement
  target_size = c(150, 150), # Redimensionne toutes les images à 150 × 150
  batch_size = 20, 
  class_mode = "binary" # Comme vous utilisez la perte binary_crossentropy, vous avez besoin d'étiquettes binaires.
  )
  return(dt_generator)
}
dt="newdata"
# Chargement du modèle
isep2 = load_model_hdf5("ISEP2_cats_Vs_dogs")
# Prédiction
images = pre_trait(dt)
images_pred = isep2 %>% predict(images) %>% `>`(0.5) %>% k_cast("int32")
images_pred
