from django.urls import path
from . import views

urlpatterns = [
    path("statball/", views.statball, name="home"),
    path("", views.statball, name="home"),
    path("signup/", views.signup, name="signup"),
    path("login/", views.login, name="login"),
    path("logout/", views.logout, name="logout"),
]