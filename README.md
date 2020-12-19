# ImageClassifierHW
Classification of the Cifar dataset using Pytorch

The classification question required me to use Pytorch to train a
classifier on a Cifar dataset. The overall goal was to classify 10000 images into ​ X ​ categories
using a convolutional neural network in Pytorch. After implementing the code, I could adjust the
parameters in order to improve the accuracy. The parameters used in the classification part
would be saved to be used for tracking in the next step. I unfortunately did not have much time
to play with the parameters, but I’ve listed them below along with the accuracy on each class of
images as well as the whole.

Training arguments for the classifier:
● Batch size: 4
● Learning rate: 0.001
● Optimization method: Stochastic Gradient descent
● Epoch number: 8

Classification testing accuracy:
● Overall: 66%
● Plane: 66%
● Car: 79%
● Bird: 44%
● Cat: 56%
● Deer: 67%
● Dog: 47%
● Frog: 76%
● Horse: 68%
● Ship: 78%
● Truck: 75%
