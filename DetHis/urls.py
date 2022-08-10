from django.urls import path
from .Vistas.perceptron import Perceptron
from .Vistas.convolutional import Convolutional
from .Vistas.knn import KNN

urlpatterns = [
    path('perceptron/', Perceptron.as_view()),
    path('convolutional/', Convolutional.as_view()),
    path('knn/', KNN.as_view()),
]

