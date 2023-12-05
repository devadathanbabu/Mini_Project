# Mini_Project
Artist identification from paintings using cnn

"Artvisionary: Artist Identification from Paintings using CNN" is a project aimed at leveraging Convolutional Neural Networks (CNNs) to recognize artists from their paintings. The project utilizes a dataset containing paintings from various artists, focusing on those with a significant number of artworks.

Initially, it analyzes the dataset to identify artists with over 329 paintings, preparing a specific subset for classification. The code performs data augmentation using ImageDataGenerator to enhance the dataset's diversity and prevent overfitting. It then sets up training and validation data generators for the CNN model.

For the model architecture, it adopts a pre-trained ResNet50 network, fine-tuning it by adding global average pooling layers, dense layers, batch normalization, and dropout to prevent overfitting. The model is compiled using the Adam optimizer and categorical cross-entropy loss function, considering the class weights for imbalanced data.

The training process involves a learning rate scheduler and early stopping to optimize the model's performance and prevent overfitting. After training the model, it allows users to input a painting image for artist prediction. It loads the image, processes it, and uses the trained model to predict the artist, displaying the prediction probability and the artist's name.

This project essentially demonstrates the application of deep learning techniques, specifically CNNs, to identify artists based on their painting styles, showcasing the potential of AI in art analysis and identification.
