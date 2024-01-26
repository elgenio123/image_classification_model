#import tensorflow as tf
from dataset import train_test_data
from model import create_model

def main():

    cifar_10_folder_path = 'cifar-10-batches-py'
    x_train, y_train, x_test, y_test = train_test_data(cifar_10_folder_path)

    # x_train and x_test are the training and testing image data, respectively.
    # y_train and y_test are the corresponding labels.

    # Print the shape of the loaded data
    print("Training data shape:", x_train.shape)
    print("Training labels shape:", y_train.shape)
    print("Testing data shape:", x_test.shape)
    print("Testing labels shape:", y_test.shape)

    model = create_model(x_train, y_train, x_test, y_test)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test)

    # Get predictions (probabilities) for the test set
    predictions = model.predict(x_test)
    
    print("Probabilities for the first test sample:", predictions[0])
    print(f'Test accuracy: {test_acc}')
    print(f'Test loss: {test_loss}')

if __name__ == "__main__":
    main()