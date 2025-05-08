from django.urls import path
from . import views

urlpatterns = [
    path("statball/", views.statball, name="home")
]