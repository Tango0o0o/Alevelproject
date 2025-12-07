from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.template import loader
from .forms import SignUpForm, PredictPlayerPerformanceForm, LoginForm, SimilarPlayersForm, PredictMatchOutcomeForm
from .models import User
import bcrypt
import pandas as pd
import numpy as np
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
from .learning_models import PredictPlayerPerformance, PredictOutcome

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
        players_stats = []
        this_season_id = 25583

        if new_point_id == False or target_team_id == False: # If there false, theres an error, most likely no results found.
            return render(req, "analysis\similar-players.html", 
            {
            "form": form,
            "logged_in" : is_logged_in(req),
            "error" : "Please enter valid player and team names."
            })
            
            

        get_team = f"https://api.sportmonks.com/v3/football/squads/teams/{target_team_id}?api_token={YOUR_TOKEN}"

        squad = requests.get(get_team).json() # Getting the players in the squad 



        for player in squad["data"]:
            player_id = player["player_id"] # for each player, returng the player id
            player_stats_url =  f"https://api.sportmonks.com/v3/football/players/{player_id}?api_token={YOUR_TOKEN}&include=statistics.details&filters=playerstatisticSeasons:{this_season_id}" #returning the statistics for each player filtered on this season

            player_stats = requests.get(player_stats_url).json() # return that players stats
            players_stats.append(player_stats["data"]) # add the stats to an array
            
        with open("temp.json", "w") as f:
            f.write(json.dumps(players_stats, indent=4)) # save
        
        
        with open("temp.json", "r") as f:
            player_data = json.load(f)

        # binary search
        def type_search(array, item):

            if not array:
                return "Not found"
            
            middle = round((len(array)-1)/2) # middle index

            if array[middle]["id"] == item: # check if middle is the item

                return array[middle] # if so the return 
            
            else:

                if array[middle]["id"] > item: # if item is less than, remove 2nd half of list
                    array = array[0:middle]

                else: # oppposite
                    array = array[middle+1:len(array)] # end is exlusive.
                
            if len(array) == 1: # if only 1 item, check if its the item, if not then yeah
                # could remove this is you use ceil instead of rounding
                if array[0]["id"] != item:
                    return "Not found"
                else:
                    return array[0]

            return type_search(array, item)

        # returning type information by id
        def find_type_by_id(id):

            with open("types.json", "r") as f:
                types_data = json.load(f)

            result = type_search(types_data, id)
            return result
            
        def create_player_dataframe(player_data):
            df = pd.DataFrame() 

            for player in player_data: # every player
                
                # not sure which to use, will just leave both here for now
                player_id = player["id"] # getting the player id
                player_name = player["name"] # getting the player name

                for detail in player["statistics"][0]["details"]: # get their statistic details, return the type id and get info
                    
                    type_id = detail["type_id"] # getting the type id

                
                    type_info = find_type_by_id(id=type_id) # returning the info of the type
                    type_name = type_info["name"] # getting the types name
                    
                    
                    value = detail["value"].get("total", 0) # getting the value associated with that type

                    
                    
                    if type_name not in df.columns: # check if column doesn't exist...
                        df[type_name] = 0.0 # add column to df if doesn't exist, default value to 0

                    df.loc[player_id, type_name] = float(value) # set the players stat to the value, using the player id as the index, in the colums typename
                
            df = df.fillna(0) # fill missing values with 0, doing this as if the api returns none, it isn't an unknown, the stat doesn't exist for that player as they have none
            return df


        # creating the dataframe...
        # create df✅ -> go statistics ✅ -> details -> type_id✅ -> fetch stat name✅ -> add column if it doesn't exist✅ -> populate with player value✅

        # PlayerName Stat1 Stat2 ... Statn

        # getting player by id or name from api
        def get_player(player_id=None, name=None):
            if player_id != None: # if no id, then use name instead
                url = f"https://api.sportmonks.com/v3/football/players/{player_id}?api_token={YOUR_TOKEN}&include=statistics.details&filters=playerstatisticSeasons:25583"
                req = requests.get(url).json()
            
            return req

        # adding player to df
        def add_player_to_df(df, player_id):
                
            player_data = get_player(player_id=player_id)
            
            df.loc[player_id] = 0.0 # defaul to 0 of existing stats


            for detail in player_data["data"]["statistics"][0]["details"]: # get their statistic details, return the type id and get info
                
                type_id = detail["type_id"] # getting the type id

            
                type_info = find_type_by_id(id=type_id) # returning the info of the type
                type_name = type_info["name"] # getting the types name
                
                
                value = detail["value"].get("total", 0) # getting the value associated with that type


                
                if type_name not in df.columns: # check if column doesn't exist...
                    print("No column")
                else:

                    df.loc[player_id, type_name] = float(value) # set the players stat to the value, using the player id as the index, in the colums typename
            return df


        df = create_player_dataframe(player_data) # creating df


        # KNN classifier
        df = create_player_dataframe(player_data) # creating df

        # Standard scaler formula
        def standard_scaler(feature, point):
            std = np.std(feature)
            if std == 0: # prevent invalid division
                return 0  

            return ( point - np.mean(feature)  ) / std

        points = {} # dictionary of points

        for player_id, group in df.groupby(df.index):# for each playerd_id in the indexes of the df and the values of it...
            points[player_id] = group.values.tolist() # set the key of the player id to the values of it in the df, (in a list)

        df = add_player_to_df(df, player_id=new_point_id) # creating new point in the df, becuase only certain columns are needed
    
        new_point = df.loc[new_point_id].values.tolist() # features of the point/player

        def euclidean_distance(p, q): # 2 points
            return np.sqrt(np.sum((np.array(p) - np.array(q))** 2))

        # My KNN model
        class KNearestNeighbours:

            def __init__(self, k=3): # k is the number of similar players to find
                self.k = k
                self.points = None
            
            def fit(self, points): # in KNN, fitting the data is simply just having points
                self.points = points
            
            def closest_player(self, new_point):
                distances = []


                for player in self.points: # for each key (player id) in the points dict
                    for point in self.points[player]: # for each value in the dict off the keys
            
                        
                        distance = euclidean_distance(point, new_point) # calculate distance
                        distances.append([distance, player]) # add to distances for sorting + classification

                # closest_player = sorted(distances)[:1][0][1] # sort it by the distance, then return the player_id inside the first array

                closest_players = [player[1] for player in sorted(distances)[:self.k]]

                return closest_players
            
        clf = KNearestNeighbours()
        clf.fit(points) # Adding points to the class

        closest_players = clf.closest_player(new_point) # these are the player ids

        # Every point in the dataframe is scaled.
        # for j in range(0, len(df.columns)):
        #     for i in range(0, len(df.index)):
        #         df.iloc[i, j] = standard_scaler(df.iloc[:,j] ,df.iloc[i, j])

        # The points need to be dimension reduced, to do this it has to be a numpy array, hence the conversion.

        points_list = [df.loc[closest_players[i]].tolist() for i in range(0, len(closest_players))]
        points_list.append(df.loc[new_point_id].tolist())

        # Get the closest points values and convert to numpy array
        closest_points = np.array(points_list)
        #new_point

        closest_points_reduced = TSNE(n_components=2, perplexity=2.0).fit_transform(closest_points) # reducing dimensions to 2


        plt.scatter(x=[i[0] for i in closest_points_reduced], y=[i[1] for i in closest_points_reduced])

        with open("temp.json", "r") as f: # getting image paths for each player from database
            player_data = json.load(f)

            images = [player["image_path"] for player in player_data if player["id"] in closest_players]
            images.append(

                requests.get(f"https://api.sportmonks.com/v3/football/players/{new_point_id}?api_token={YOUR_TOKEN}").json()["data"]["image_path"]

            )
 

        def load_image(img_path):
            if img_path.startswith("http"): # If the image is a url
                with urllib.request.urlopen(img_path) as url: # Open it,
                    img = Image.open(BytesIO(url.read())) # And get the image as bytes
                return np.array(img) # Then convert to a numpy array
            else:
                return mpimg.imread(img_path) # probably never but if local load normally

        for point, img_path in zip(closest_points_reduced, images):
            img = load_image(img_path) # load image
            imgbox = OffsetImage(img, zoom=0.15) # Container for image to display
            ab = AnnotationBbox(imgbox, (point[0], point[1]), frameon=False) # Then annotate it at the coordinates
            plt.gca().add_artist(ab) # Then add it to the plot

        # plt.show()

        categories = df.columns # The categories for the radar chart

        fig = go.Figure() # Newfigure/annotation to the graph
        
        flat_cols = []

        for i in range(0, closest_points.shape[1]):
            mx = max(closest_points[:,i])
            mi = min(closest_points[:,i])


            for j in range(0, closest_points.shape[0]):
                
                if mx == mi:
                    if i not in flat_cols:
                        flat_cols.append(i)
                else:
                    closest_points[j, i] = ( closest_points[j, i] - mi) / (mx - mi)

        closest_points = np.delete(closest_points, flat_cols, axis=1)


        closest_players.append(new_point_id)

        for i in range(0, len(closest_players)): # Froe each player,
            fig.add_trace(go.Scatterpolar( # Add a radar trace

                r=closest_points[i], # In which the radial axis is the values of each feature
                theta=categories, # And the other axis is the name of the features themselves 
                fill="toself", # Connects the endpoints of the trace into a closed shape
                name=str(requests.get(f"https://api.sportmonks.com/v3/football/players/{closest_players[i]}?api_token={YOUR_TOKEN}").json()["data"]["display_name"]) # And then the name of the trace
                

            ))


        # Apply the changes to the figure
        fig.update_layout(

            polar=dict(
                radialaxis=dict(
                    visible = True,
                    range=[0,1] 
                )
            ),
            title=dict(text="Player comparison based on similarity score 0-1")

        )
        
        
        fig.show() # Show figure in new tab
        time_img = str(round(time.time())) # get current time that analysis took place
        filename = f"{req.session.get("user_id")}-{time_img}" # create a filename

        image_path = f"Main/static/images/analysis/similar_players/{filename}.png" # path to write image to in the project
        fig.write_image(image_path) # write to that path
        django_path = f"images/analysis/similar_players/{filename}.png" # This will be the path that django template will use to get the static image for html 
    
        
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


        # put it on the page
        return render(req, "analysis\submit_outcome_stats.html", 
            {
            "form": req.POST,
            "logged_in" : is_logged_in(req),
          #  "django_path" : django_path, # bar chart image path to display on page
            "cleaned_data_dict": cleaned_data_dict, 
            "value" : final_index, # final index, decides whether it was a win draw or loss
            "team_name" : pred_outcome.team_name 
            # Passing through the column names & values back into the page so the user doesn't have to reenter every value if they want to adjust parameters
            })
    else: # if not a form input, user shouldn't be here
        return redirect("predict-match")


