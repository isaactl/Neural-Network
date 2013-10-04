#include <QCoreApplication>
#include "neuralnetwork.h"

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    // create a neural network
    NeuralNetwork ann(9,15,9,1);

    ann.trainNetwork();
    ann.enableLogging("result.txt");
    a.exit();

    return 0;
}
