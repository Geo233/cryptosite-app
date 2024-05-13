
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name="home"),
    path('prices/', views.prices, name="prices"),
    path('predictions/', views.predictions, name="predictions"),
    path('charts/', views.charts, name="charts"),




]
