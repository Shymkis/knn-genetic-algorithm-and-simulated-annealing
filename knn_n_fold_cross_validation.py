# Joe Shymanski
# Project 1: Adaptation (Nearest Neighbor)
import math
import pprint
import random
import time

class Data_Table:
    def __init__(self, init_data = None):
        if init_data is None:
            init_data = []
        self.data_points = init_data

    def get_points_with_label(self, label):
        return [x for x in self.data_points if x.label == label]

    def add_datum(self, datum):
        self.data_points.append(datum)

    def add_data(self, data):
        self.data_points = self.data_points + data

    def random_shuffle(self):
        random.shuffle(self.data_points)

    def normalize(self):
        for i in range(len(self.data_points[0].point)):
            mini = math.inf
            maxi = -math.inf
            for data_point in self.data_points:
                point = data_point.point
                if point[i] < mini: mini = point[i]
                if point[i] > maxi: maxi = point[i]
            for data_point in self.data_points:
                point = data_point.point
                point = list(point)
                point[i] = (point[i] - mini) / (maxi - mini)
                point = tuple(point)
                data_point.point = point

    def print_table(self):
        for data_point in self.data_points:
            data_point.print_data()

    def print_points(self, label = None):
        for data_point in self.data_points:
            if not label or label and data_point.label == label:
                data_point.print_point()


class Data_Point:
    def __init__(self, label, point, name):
        self.label = label
        self.point = point
        self.name = name

    def __lt__(self, other):
        return self.name < other.name

    def print_data(self):
        print(self.label, self.point, self.name)

    def print_point(self):
        print(self.point)


def n_folds(N, data_table):
    folds = []
    for x in range(N):
        train_data = []
        test_data = []
        for i, data_point in enumerate(data_table.data_points):
            # Every Nth point, starting from x, is stored in test set
            if i % N != x:
                train_data.append(data_point)
            # All other points are stored in train set
            else:
                test_data.append(data_point)
        folds.append({"train": Data_Table(train_data), "test": Data_Table(test_data)})
    return folds

def train_knn_model(k, train_data_table, labels, store_all):
    model_data_table = Data_Table()
    if store_all:
        model_data_table.add_data(train_data_table.data_points)
    else:
        # Add first k random points of each label to model
        for label in labels:
            l = train_data_table.get_points_with_label(label)
            model_data_table.add_data(l[:k])

        # Then iteratively classify each training point and store the errors
        for data_point in train_data_table.data_points:
            prediction = knn(k, model_data_table.data_points, [data_point])[0]
            if prediction != data_point.label:
                model_data_table.add_datum(data_point)
    return model_data_table

def knn(k, train_data, test_data):
    test_labels = []
    for test_datum in test_data:
        neighbors = []
        for train_datum in train_data:
            dist = math.dist(test_datum.point, train_datum.point)

            # If neighbors list not yet full, append data and sort list
            if len(neighbors) < k:
                neighbors.append((dist, train_datum))
                neighbors.sort()
            # If neighbors list is full, pop farthest neighbor and compare dist
            else:
                farthest_neighbor = neighbors.pop()
                if dist < farthest_neighbor[0]:
                    neighbors.append((dist, train_datum))
                    neighbors.sort()
                else:
                    neighbors.append(farthest_neighbor)

        # Calculate winning label
        labels = {}
        for neighbor in neighbors:
            data_point = neighbor[1]
            if data_point.label not in labels:
                labels[data_point.label] = 0
            labels[data_point.label] += 1
        test_labels.append(max(labels, key=labels.get))

    # Return predictions
    return test_labels

def prediction_accuracy(predictions, labeled_points):
    correct = 0
    for prediction, data_point in zip(predictions, labeled_points):
        if prediction == data_point.label:
            correct += 1
    return correct / len(predictions) * 100

def main(k, N, store_all, normalize, file_name):
    # Start timer
    start_total = time.time()

    # Read file and save entries in a data table
    print("k = " + str(k) + ", N = " + str(N) + ", Store all data = " + str(store_all) +
        ", Normalize = " + str(normalize) + ", File = " + file_name)
    train_file = open(file_name, "r")
    lines = train_file.readlines()
    if "accent" in file_name:
        lines = lines[1:]
    data_table = Data_Table()
    labels = set()
    for line in lines:
        line = line[:-1]
        values = line.replace("\"", "").replace(",", " ").split(" ")
        label = values[0]
        labels.add(label)
        point = (float(values[1]), float(values[2]))
        name = values[3]
        if "leaf" in file_name:
            point = (float(values[2]), float(values[3]), float(values[4]), float(values[5]),
                float(values[6]), float(values[7]), float(values[8]), float(values[9]),
                float(values[10]), float(values[11]), float(values[12]), float(values[13]),
                float(values[14]), float(values[15]))
            name = label + "-" + values[1]
        elif "accent" in file_name:
            point = (float(values[1]), float(values[2]), float(values[3]), float(values[4]),
                float(values[5]), float(values[6]), float(values[7]), float(values[8]),
                float(values[9]), float(values[10]), float(values[11]), float(values[12]))
            name = label
        data_point = Data_Point(label, point, name)
        data_table.add_datum(data_point)

    # Shuffle data in data table
    data_table.random_shuffle()

    # Normalize data points in data table if specified
    if normalize:
        data_table.normalize()

    # Generate N blocks for N-fold cross validation
    folds = n_folds(N, data_table)

    # Iterate over each fold and run KNN
    train_accuracy = test_accuracy = 0
    for fold in folds:
        train_data_table = fold["train"]
        test_data_table = fold["test"]

        # "Store All" vs. "Store Errors" training methods
        model_data_table = train_knn_model(k, train_data_table, labels, store_all)

        # Make predictions on training set and testing set independently
        train_predictions = knn(k, model_data_table.data_points, train_data_table.data_points)
        test_predictions = knn(k, model_data_table.data_points, test_data_table.data_points)

        # Calculate accuracies for both sets of predictions
        train_accuracy += prediction_accuracy(train_predictions, train_data_table.data_points)
        test_accuracy += prediction_accuracy(test_predictions, test_data_table.data_points)

    # Average accuracies of all test runs
    print("Average training set accuracy: " + str(round(train_accuracy / N, 2)) + "%")
    print("Average testing set accuracy: " + str(round(test_accuracy / N, 2)) + "%")

    # Stop timer, report time elapsed
    end_total = time.time()
    print(round(end_total - start_total, 2), "seconds elapsed in total")
    print()

if __name__ == "__main__":
    main(3, 5, True, True, "train-file")
    main(3, 5, True, False, "train-file")
    main(3, 5, False, True, "train-file")
    main(3, 5, False, False, "train-file")
    main(3, 5, True, True, "leaf/leaf.csv")
    main(3, 5, True, False, "leaf/leaf.csv")
    main(3, 5, False, True, "leaf/leaf.csv")
    main(3, 5, False, False, "leaf/leaf.csv")
    main(3, 5, True, True, "accent-recognition-mfcc--1/accent-mfcc-data-1.csv")
    main(3, 5, True, False, "accent-recognition-mfcc--1/accent-mfcc-data-1.csv")
    main(3, 5, False, True, "accent-recognition-mfcc--1/accent-mfcc-data-1.csv")
    main(3, 5, False, False, "accent-recognition-mfcc--1/accent-mfcc-data-1.csv")
