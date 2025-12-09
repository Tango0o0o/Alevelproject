from django.urls import path
from . import views

urlpatterns = [
    path("statball/", views.statball, name="home"),
    path("", views.statball, name="home"),
    path("signup/", views.signup, name="signup"),
    path("login/", views.login, name="login"),
    path("predict-match/", views.pred_match, name="predict-match"),
    path("predict-match-result/", views.pred_match_result, name="predict-match-result"),
    path("logout/", views.logout, name="logout"),
    path("similar-players/", views.sim_players, name="similar-players"),
    path("predict-player/", views.pred_player, name="predict-player"), 
    path("predict-performance-result/", views.pred_performance_result, name="predict-performance-result"),

    path("previous-analysis/", views.prev_analysis, name="previous-analysis"),
    
]