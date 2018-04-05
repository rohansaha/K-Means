"""==============================================================
 COURSE:	    CSC 635, Homework 3
 PROGRAMMER:	Rohan Saha  ID on trace: Rohan2728
 DATE:	        4/5/2018
 DESCRIPTION:   To implement a K-Means algorithm, using
                synthetic_2D.txt dataset.
 FILES:	        hw3.py
 DATASET:       synthetic_2D.txt
 =============================================================="""

# ---------------------------------Imports--------------------------------------
from csv import reader
from math import pow, sqrt
import random
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------


#---------------------------------Variables------------------------------------

# global variables
fig_size = 16 / 2.2, 9 / 2.2
fig, ax = plt.subplots(figsize=fig_size)
plt.title('K Means Clusters')
ax.set_facecolor('whitesmoke')

#------------------------------------------------------------------------------

# ---------------------------------Functions------------------------------------

'''Draw scatter plots '''
def plot_scatter(data,isSubplots = False, label=None):
    clusters = set(map(lambda x: x[2], data))
    newDataset = [[v for v in data if v[2] == c ] for c in clusters]
    colors = ['c', 'm', 'y']
    i= 0
    for cluster in newDataset:
        f_x, f_y, f_cluster = zip(*cluster)
        if isSubplots:
            label = 'Cluster ' + str(set(f_cluster))
            ax.scatter(f_x, f_y, c=colors[i], label=label)
            ax.legend()

        else:
            plt.title('Cluster Label ' + str(label))
            c_label = 'Cluster ' + str(set(f_cluster))
            plt.scatter(f_x, f_y, c=colors[i], label=c_label)
            plt.legend()
        i += 1


'''Normalizes the data set using Min-Max Normalization'''
def normalize(data):
    c_x = []
    c_y = []
    c_label = []

    # Type cast x,y and label values into float and int respectively
    list(map(lambda x: (c_x.append(float(x[0])), c_y.append(float(x[1])), c_label.append(int(x[2]))), data))

    min_x_Value = min(c_x)
    max_x_Value = max(c_x)
    min_y_Value = min(c_y)
    max_y_Value = max(c_y)

    result = list(map(lambda i: ((c_x[i] - min_x_Value) / (max_x_Value - min_x_Value),
                                 (c_y[i] - min_y_Value) / (max_y_Value - min_y_Value),
                                 c_label[i]), range(len(c_x))))

    # Scatter Plot to Visualize all the data points
    plt.xlabel('X')
    plt.ylabel('Y')

    plot_scatter(result, True)

    # Non - Normalized results
    # result = list(map(lambda i: (c_x[i],c_y[i],c_label[i]), range(len(c_x))))
    return result


'''Calculates the distance between two attribute values'''
def eucildean(means, dataPoint, isError=False):
    distance = []
    if isError:
        list(map(lambda m, d: distance.append(sqrt(pow(d[0] - m[0], 2) + pow(d[1] - m[1], 2))), means, dataPoint))
    else:
        d_x, d_y, d_c = dataPoint
        list(map(lambda m: distance.append(sqrt(pow(d_x - m[0], 2) + pow(d_y - m[1], 2))), means))
    return distance


'''Reads the data from text file and normalizes the dataset'''
def dataPreprocessing(filename):
    with open(filename, 'r') as file:
        data = list(reader(file, delimiter=' '))
    data = normalize(data)
    return data


'''Calculates new mean values'''
def calculateMean(cluster):
    m_x, m_y, m_c = zip(*cluster)
    mean_x = sum(m_x) / len(m_x)
    mean_y = sum(m_y) / len(m_y)
    return mean_x, mean_y


'''Gets Random data points as initial mean'''
def getInitialMean(k, D):
    random.seed(seed_n)
    random.shuffle(D)
    mean = random.sample(D, k)
    for i in range(k):
        print('mean[', i, '] is ', mean[i])

    # Plot to show the new mean values in scatter plot
    c_x, c_y, c = zip(*mean)
    ax.plot(c_x, c_y, 'k*', markersize=10, label='Initial Mean')
    ax.legend()
    return mean


def KMeans(D, k):
    print('Initial k means are')
    means = getInitialMean(k, D)
    means_old = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
    squaredError = min(eucildean(means, means_old, True))
    while squaredError != 0:
        K = [[], [], []]
        for i in range(len(D)):
            distances = eucildean(means, D[i])
            minIndex = distances.index(min(distances))
            K[minIndex].append(D[i])

        for i in range(k):
            means_old[i] = means[i]
            means[i] = calculateMean(K[i])

        c_x, c_y = zip(*means)
        ax.plot(c_x, c_y, 'bX-', markersize=10, label='Final Mean')
        ax.legend()
        plt.show()
        squaredError = min(eucildean(means, means_old, True))

    return K

'''Gets the Accuracy result'''
def Accuracy(clusters):
    correct = 0
    size = 0
    for i in range(len(clusters)):
        print()
        print('===========================================================')
        print('Cluster ', i)
        size += len(clusters[i])
        print('Size of cluster ', i, 'is ', len(clusters[i]))

        x, y, c_labels = zip(*clusters[i])
        label = max(c_labels, key=c_labels.count)


        plot_scatter(clusters[i], label=label)
        plt.show()

        print('Cluster label:', label)

        misses = 0
        for l in c_labels:
            if l == label:
                correct += 1
            else:
                misses += 1
        print('Number of objects misclustered in this cluster is ', misses)
        print()
        temp = list(map(lambda d: print(d), clusters[i]))

    accuracy = round((correct/size)*100, 2)
    print()
    print('Overall Accuracy Rate: ', accuracy, '%')


# ------------------------------------------------------------------------------

# ---------------------------------Program Main---------------------------------

if __name__ == '__main__':
    filename = '.\\dataset\\synthetic_2D.txt'
    data = dataPreprocessing(filename)
    k = 3
    seed_n = 21
    Clusters = KMeans(D=data, k=k)
    Accuracy(Clusters)

# ---------------------------------End of Program-------------------------------