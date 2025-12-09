from django.db import models
import datetime

# Create your models here.

class User(models.Model):
    email = models.CharField(name="email", max_length=99)
    password = models.CharField(name="password")
    # ID automatically created

    def __str__(self):
        return self.email


class AnalysisType(models.IntegerChoices): # choices for which type 
    SIM_PLAYERS = 0, "Similar players"
    PRED_PERFORMANCE = 1, "Predict player performance"
    PRED_OUTCOME = 2, "Predict team outcome"

class Analysis(models.Model):

    # fields for the analysis, ones with null are specific to some but not all of the different analysistypes
    name = models.CharField(name="name", max_length=500) # make migrations
    image_path = models.CharField(name="image_path", null=True)
    user_id = models.IntegerField(name="user_id")
    data = models.JSONField(default=dict, null=True)
    type = models.IntegerField(
        choices=AnalysisType.choices
    )
    final_index = models.IntegerField(name="final_index", null=True)
    team_name = models.TextField(name="team_name", null=True)
    fixture_name = models.TextField(name="fixture_name", null=True)