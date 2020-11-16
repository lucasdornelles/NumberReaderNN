from src.NumberReader import NumberReader
from src.MNISTParser import read_dataset
import sys
import os


# 784


def test_number_reader(number_reader, test_dataset):
    dataset = list(test_dataset)
    n_items = len(dataset)
    times_correct = 0.
    total_cost = 0.
    for i in range(n_items):
        values = dataset[i][0]
        label = dataset[i][1]
        result = number_reader.solve_number_image(values)
        cost = number_reader.calculate_cost_function(label)
        times_correct += 1 if result == label else 0
        total_cost = total_cost + cost
    precision = times_correct / n_items
    mean_cost = total_cost / n_items
    print("precision, mean cost")
    print((precision, mean_cost))


if __name__ == '__main__':

    number_reader = NumberReader(784, 4, 16)

    if len(sys.argv) >= 2:
        if len(sys.argv) == 3 and sys.argv[1] == "train":
            if not sys.argv[2].isdigit():
                print("use ' Main.py retrain ' to train a neural network with 20 epochs, "
                      "' Main.py test' to tested saved neural network or "
                      "' Main.py train #n ' to further train a saved neural network with n epochs")
                exit(1)
            if not os.path.isfile("NumberReaderData.yaml"):
                print("Missing NumberReaderData.yaml file")
                exit(1)

            with open("NumberReaderData.yaml") as file:
                success = number_reader.load_number_reader(file)
                if not success:
                    print("NumberReaderData.yaml file is not compatible")
                    exit(1)
            for i in range(int(sys.argv[2])):
                training_dataset = read_dataset("resources/train-images.idx3-ubyte", "resources/train-labels.idx1-ubyte")
                number_reader.batch_training(training_dataset, 100)
                test_dataset = read_dataset("resources/t10k-images.idx3-ubyte", "resources/t10k-labels.idx1-ubyte")
                test_number_reader(number_reader, test_dataset)

            with open("NumberReaderData.yaml", "w") as file:
                number_reader.save_number_reader(file)

        elif sys.argv[1] == "retrain":

            # Train 20 epochs and test on every epoch

            for i in range(20):
                training_dataset = read_dataset("resources/train-images.idx3-ubyte", "resources/train-labels.idx1-ubyte")
                number_reader.batch_training(training_dataset, 100)
                test_dataset = read_dataset("resources/t10k-images.idx3-ubyte", "resources/t10k-labels.idx1-ubyte")
                test_number_reader(number_reader, test_dataset)

            with open("NumberReaderData.yaml", "w") as file:
                number_reader.save_number_reader(file)

        elif sys.argv[1] == "test":

            if not os.path.isfile("NumberReaderData.yaml"):
                print("Missing NumberReaderData.yaml file")
                exit(1)

            with open("NumberReaderData.yaml") as file:
                success = number_reader.load_number_reader(file)
                if not success:
                    print("NumberReaderData.yaml file is not compatible")
                    exit(1)

            test_dataset = read_dataset("resources/t10k-images.idx3-ubyte", "resources/t10k-labels.idx1-ubyte")
            test_number_reader(number_reader, test_dataset)

        else:
            print("use ' Main.py retrain ' to train a neural network with 20 epochs, "
                  "' Main.py test' to tested saved neural network or "
                  "' Main.py train #n ' to further train a saved neural network with n epochs")
            exit(1)

    else:
        print("use ' Main.py retrain ' to train a neural network with 20 epochs, "
              "' Main.py test' to tested saved neural network or "
              "' Main.py train #n ' to further train a saved neural network with n epochs")
        exit(1)