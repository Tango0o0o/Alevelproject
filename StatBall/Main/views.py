from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.template import loader
from .forms import SignUpForm, PredictPlayerPerformanceForm, LoginForm, SimilarPlayersForm, PredictMatchOutcomeForm
from .models import User, Analysis
import bcrypt
import pandas as pd
import numpy as np
import datetime
import requests, json
import csv
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
from PIL import Image
import urllib.request
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px
import time
from .learning_models import PredictPlayerPerformance, PredictOutcome, SimilarPlayers

YOUR_TOKEN = "p7pnma41hZ54JY3pwMd1GXh3cWykgQYiqUzdVcOlxVcLsvfXblU5B4oTT76M"

pred_player_models_temp = {} # holds the machine learning models for each user
pred_outcome_models_temp = {} # holds the machine learning models for each user
# Extra functions

# Checks if there is a user logged in
def is_logged_in(req):
    if req.session.get("user_id"): # Checks if a user id exists in the session
        return True
    return False


# The actual views that lead to pages
def statball(req):
    # Passing through the required variables.
    return render(req, 
                  "default.html", 
                  { # This is a dictionary passing in variabled into the template
                      "logged_in" : is_logged_in(req),  # Boolean
                      "user" : req.session.get("user_id"),
                    }
                  ) 

# Here, validation and user creation takes place
def signup(req):

    logged_in = is_logged_in(req)

    if logged_in: # if logged in, you can't sign up
       return redirect("home")

    form = SignUpForm()
    email_msg = ""
    password_msg = ""

    # Post meaning data is being sent to the server; i.e. from the user
    if req.method == "POST":
        form = SignUpForm(req.POST) #Filling the form with POST data, (user entered)
        
        valid_email = form.is_valid_email()
        
        if valid_email != True:
            email_msg = valid_email
        
        valid_pass = form.is_valid_password()

        if valid_pass != True:
            password_msg = valid_pass

       
        # The salt is a random value added to the password before hashing
        salt = bcrypt.gensalt()
        hashed_password = bcrypt.hashpw(form.get_password().encode(), salt)

        if valid_email == True and valid_pass == True: # All good, create and save the user to the DB, log them in, redirect to home
            user = User(email=form.get_email(),password=hashed_password)
            user.save()
            req.session.create()
            req.session['user_id'] = user.id
            return redirect("home")

    
    return render(req, "accounts/signup.html", 
        {
            "form":form, 
            "email_msg":email_msg, 
            "password_msg":password_msg,
            "logged_in" : logged_in
        })

# Authenticates the user and creates a new session
def login(req):

    logged_in = is_logged_in(req)

    if logged_in:
        return redirect("home")

    login_form = LoginForm()
    message = ""

    if req.method == "POST":

        login_form = LoginForm(req.POST)
        validation = login_form.authenticate() # Check if credentials match in the database

        # validation[0] is a bool
        if validation[0] == True:

            req.session.create()
            req.session['user_id'] = validation[1].id # validation[1] is the user object
            return redirect("home")
        
        else:
            message = "No matching account found"

    return render(req, "accounts/login.html", 
        {"login_form": login_form, 
         "message":message,
         "logged_in" : logged_in
         }) # returning the webpage

# Logs the user out by clearing the session
def logout(req):
    
    if is_logged_in(req): # Only execute the process of the users is actually logged in
        req.session.flush()
    
    return redirect("home")

def sim_players(req):
    if req.method == "GET":
        form = SimilarPlayersForm() # Input for the user
         
        return render(req, "analysis\similar-players.html", {"form": form, "logged_in" : is_logged_in(req)}) # show the page

    else:
        form = SimilarPlayersForm(req.POST)
        new_point_id = form.get_player_id() # The new players ID
        target_team_id = form.get_team_id() # The teams players to compare the reference player to


        if new_point_id == False or target_team_id == False: # If there false, theres an error, most likely no results found.
            return render(req, "analysis\similar-players.html", 
            {
            "form": form,
            "logged_in" : is_logged_in(req),
            "error" : "Please enter valid player and team names."
            })
        

            
        
        

        sim_players = SimilarPlayers(new_point_id, target_team_id)
        sim_players.perform_analysis()
        fig = sim_players.scatter_plot()
        
        
        
        fig.show() # Show figure in new tab
        time_img = str(round(time.time())) # get current time that analysis took place
        filename = f"{req.session.get("user_id")}-{time_img}" # create a filename

        image_path = f"Main/static/images/analysis/similar_players/{filename}.png" # path to write image to in the project
        fig.write_image(image_path) # write to that path
        django_path = f"images/analysis/similar_players/{filename}.png" # This will be the path that django template will use to get the static image for html 

        save_analysis(req=req, name=f"Similar players to {sim_players.player_name} from {sim_players.team_name} {datetime.datetime.now()}", type=0, image_path=django_path)
        
        return render(req, "analysis\similar-players.html", 
            {
            "form": form,
            "logged_in" : is_logged_in(req),
            "django_path" : django_path
            })

def pred_player(req):
    
    if req.method == "GET": # if not a form, create one for the user
        form = PredictPlayerPerformanceForm()
        return render(req, "analysis\predict_performance.html", {"form": form, "logged_in" : is_logged_in(req)}) # show the page
    else:
        form = PredictPlayerPerformanceForm(req.POST) # otherwise get the form input
        player_id = form.get_player_id() # Get the new players ID

        if player_id == False: # If its false, theres an error, most likely no results found.
            return render(req, "analysis\predict_performance.html", 
            {
            "form": form,
            "logged_in" : is_logged_in(req),
            "error" : "Please enter valid player name."
            })
        else: # Otherwise
            pred_player = PredictPlayerPerformance(player_id=player_id) # Setup the ML model for this player
            pred_player.create_player_pred_model() # Create the model
            column_names = pred_player.columns # Get the column names of the stats the model uses

            pred_player_models_temp[req.session.get("user_id")] = pred_player # save the model into a dictionary so it can be used across different views
            return render(req, "analysis\submit_stats.html", # return the page
                {
                    "logged_in" : is_logged_in(req),
                    "column_names": column_names,
                })

   # model = PredictPlayerPerformance(player_id=)

def pred_performance_result(req):

    if req.method == "POST": # if a form input
        cleaned_data_dict = {}

        pred_player = pred_player_models_temp[req.session.get("user_id")] # get the ML model for the user

        for key, value in req.POST.items():
            if key == "csrfmiddlewaretoken" or key == "Sign up": # ignore these, they're part of the form but not for the model
                continue
            
            # set each column name to its value in the dict
            # if there is no input, assume it is 0
            if not value.strip():
                cleaned_data_dict[key] = float(0)
            else:
                cleaned_data_dict[key] = float(value)

        pred_player.predict_peformance(list(cleaned_data_dict.values())) # predict the rating
        django_path = pred_player.bar_chart(req) # get the image path of the bar chart

        
        save_analysis(req=req, type=1, name=f"Prediction for {pred_player.player_name} {datetime.datetime.now()}", data=cleaned_data_dict, image_path=django_path)

        # put it on the page
        return render(req, "analysis\submit_stats.html", 
            {
            "form": req.POST,
            "logged_in" : is_logged_in(req),
            "django_path" : django_path, # bar chart image path to display on page
            "cleaned_data_dict": cleaned_data_dict, 
            # Passing through the column names & values back into the page so the user doesn't have to reenter every value if they want to adjust parameters
            })
    else: # if not a form input, user shouldn't be here
        return redirect("predict-player")

def pred_match(req):
    if req.method == "GET":
        form = PredictMatchOutcomeForm() # Input for the user
         
        return render(req, "analysis\pred_outcome.html", {"form": form, "logged_in" : is_logged_in(req)}) # show the page
    else:
        form = PredictMatchOutcomeForm(req.POST) # otherwise get the form input
        team_id = form.get_team_id() # Get the new players ID

        if team_id == False: # If its false, theres an error, most likely no results found.
            return render(req, "analysis\pred_outcome.html", 
            {
            "form": form,
            "logged_in" : is_logged_in(req),
            "error" : "Please enter valid player name."
            })
        else: # Otherwise
            pred_outcome = PredictOutcome(team_id=team_id) # Setup the ML model for this team
            pred_outcome.get_data()
            pred_outcome.create_model() # Create the model
            column_names = pred_outcome.columns # Get the column names of the stats the model uses

            pred_outcome_models_temp[req.session.get("user_id")] = pred_outcome # save the model into a dictionary so it can be used across different views
            return render(req, "analysis\submit_outcome_stats.html", # return the page
                {
                    "logged_in" : is_logged_in(req),
                    "column_names": column_names,
                })

def pred_match_result(req):
    if req.method == "POST": # if a form input
        cleaned_data_dict = {}

        pred_outcome = pred_outcome_models_temp[req.session.get("user_id")] # get the ML model for the user
        team_name = pred_outcome.team_name
        fixture_name = pred_outcome.next_fixt_name

        for key, value in req.POST.items():
            if key == "csrfmiddlewaretoken" or key == "Sign up": # ignore these, they're part of the form but not for the model
                continue
            
            # set each column name to its value in the dict
            # if there is no input, assume it is 0
            if not value.strip():
                cleaned_data_dict[key] = float(0)
            else:
                cleaned_data_dict[key] = float(value)

        stats = list(cleaned_data_dict.values())

        final_index = pred_outcome.final_outcome([stats]) # predict the rating
        #django_path = pred_player.bar_chart(req) # get the image path of the bar chart

        save_analysis(req=req, final_index=final_index, team_name=team_name, type=2, name=f"Prediction for {pred_outcome.team_name} ({pred_outcome.next_fixt_name}) {datetime.datetime.now()}", data=cleaned_data_dict, fixture_name=fixture_name)
   
        # put it on the page
        return render(req, "analysis\submit_outcome_stats.html", 
            {
            "form": req.POST,
            "logged_in" : is_logged_in(req),
          #  "django_path" : django_path, # bar chart image path to display on page
            "cleaned_data_dict": cleaned_data_dict, 
            "value" : final_index, # final index, decides whether it was a win draw or loss
            "team_name" : team_name,
            "fixture" : fixture_name
            # Passing through the column names & values back into the page so the user doesn't have to reenter every value if they want to adjust parameters
            })
    else: # if not a form input, user shouldn't be here
        return redirect("predict-match")
    
def save_analysis(req, name, type, final_index=None, team_name=None, data=None, image_path=None, fixture_name=None):

    analysis = Analysis(name=name, user_id=int(req.session.get("user_id")),type=type, image_path=image_path, data=data, final_index=final_index, team_name=team_name, fixture_name=fixture_name)
    analysis.save()
    
def prev_analysis(req):

    if req.method == "GET": # Show a list of the users prev analysis
        user_analysis_data = Analysis.objects.filter(user_id=int(req.session.get("user_id")))

        return render(req, "analysis\prev_analysis.html", 
                {"user_analysis" : user_analysis_data,
            "logged_in" : is_logged_in(req),
                 
                 }
                 )
    else:
        # filter analysis that is by the user and matches the analysis name from the form
        user_id = req.session.get("user_id") 
        analysis_name = req.POST.get("analysis_name")
        analysis = Analysis.objects.filter(user_id=int(user_id), name=analysis_name)[0]
        

        # decide which type it is, and load the right webpage, populating with the right values.
        if analysis.type == 0:
            img_path = analysis.image_path

            return render(req, "analysis\similar-players.html", # finding similar players
            {
                "logged_in" : is_logged_in(req),
                "django_path" : img_path,
                "analysis_name" : analysis_name,
                "previous_analysis" : True
            })
        elif analysis.type == 1:
            data = analysis.data
            django_path = analysis.image_path

            return render(req, "analysis\submit_stats.html", # prediction player performance
            {
                "logged_in" : is_logged_in(req),
                "django_path" : django_path, # bar chart image path to display on page
                "analysis_name" : analysis_name,
                "cleaned_data_dict": data, 
                "previous_analysis" : True

                # Passing through the column names & values back into the page so the user doesn't have to reenter every value if they want to adjust parameters
            })
        else:

            final_index = analysis.final_index
            data = analysis.data
            team_name = analysis.team_name
            fixture_name = analysis.fixture_name

            return render(req, "analysis\submit_outcome_stats.html", # predicting team performace
            {
                "logged_in" : is_logged_in(req),
            #  "django_path" : django_path, # bar chart image path to display on page
                "cleaned_data_dict": data, 
                "value" : final_index, # final index, decides whether it was a win draw or loss
                "team_name" : team_name,
                "fixture" : fixture_name,
                "previous_analysis" : True
                # Passing through the column names & values back into the page so the user doesn't have to reenter every value if they want to adjust parameters
            })
