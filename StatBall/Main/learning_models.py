import requests
import json
import pandas as pd
from math import trunc
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import pandas as pd
import numpy as np, random
import random
import time
YOUR_TOKEN = "p7pnma41hZ54JY3pwMd1GXh3cWykgQYiqUzdVcOlxVcLsvfXblU5B4oTT76M"

class Type_finder():
    # binary search

    def __init__(self, id):
        self.id = id

    def type_search(self, array, item):

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

        return self.type_search(array, item)

    # returning type information by id
    def find_type_by_id(self):

        with open("types.json", "r") as f:
            types_data = json.load(f)

        result = self.type_search(types_data, self.id)
        return result

class PredictOutcome():

    def __init__(self, team_id, games=3):
        self.team_id = team_id
        self.team_name = None
        self.games = games
        self.stats = None
        self.mean = None
        self.std = None
        self.columns = None
        
    def get_data(self):
        team_id = self.team_id
        games = self.games # number of recent results to use for ML
        team_results = requests.get(f"https://api.sportmonks.com/v3/football/teams/{team_id}?api_token={YOUR_TOKEN}&include=latest.statistics;latest.participants").json() # Teams results
        stats = {}

        with open("temp2.json", "w") as f:
            f.write(json.dumps(team_results, indent=4))
        self.team_name = team_results["data"]["name"]
        
        loss = False
        draw = False
        win = False

        for i in range(0, games):
            
            participants = {}

            for participant in team_results["data"]["latest"][i]["participants"]: # for team in each match
                participants[participant["id"]] = participant # set the team id to the team
                if participant["id"] != team_id: # if its not the target team
                    opponent_id = participant["id"] # set the opponents id to the id

            opponent_winner = participants[opponent_id]["meta"]["winner"] # bool, whether they won or not
            team_winner = participants[team_id]["meta"]["winner"]

            if opponent_winner == team_winner: # equal means both false, meaning a draw
                if draw: # if we already have a draw result, skip
                    continue
                outcome = [0, 1, 0] # index 1 = draw
                draw = True
            elif opponent_winner:
                if loss: 
                    continue
                outcome = [1, 0, 0] # index 0 = loss
                loss = True
            else: 
                if win: 
                    continue
                outcome = [0, 0, 1] # index 2 = win
                win = True

            match = team_results["data"]["latest"][i] # Get results in the specified range
            match_stats = {} # Stats for each game
            
            if participants[team_id]["meta"]["location"] == "home": # is home stat
                is_home = 1
            else:
                is_home = 0

            # This may break if it tries to fetch it from a cup game.
            opponent_position = participants[opponent_id]["meta"]["position"] # position of the teams
            team_position = participants[team_id]["meta"]["position"] 

            # adding stuff to the stats
            match_stats["Outcome"] = outcome
            match_stats["Is Home"] = is_home
            match_stats["Team Position"] = team_position
            match_stats["Opponent Position"] = opponent_position

            for statistic in match["statistics"]: # For each stat
                

                if statistic["participant_id"] == team_id: # If it's for the target team
                    type_id = statistic["type_id"] # get type id
                    finder = Type_finder(id=type_id)
                    type_info = finder.find_type_by_id() # returning the info of the type
                    type_name = type_info["name"] # getting the types name

                    value = statistic["data"].get("value", 0) # get the value

                    match_stats[type_name] = value # Add to dict for that fixture

            
            
            stats[match["id"]] = match_stats # Set it in the dict for that game



        
        self.stats = stats # Return dict for df

        return stats

    def softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)  # stability fix
        z_exp = np.exp(z)
        return z_exp / np.sum(z_exp, axis=1, keepdims=True)

    def calc_logits(self, betas, X_b):
            logits = [] # list to store logits for every sample

            for sample in X_b: # go through each row/sample in X_b

                sample_logits = [] # store logits for this sample
                for beta in betas: # for each set of weights
                    logit = np.dot(sample, beta)  # dot product multiplication
                    sample_logits.append(logit) # save logit 

            
                logits.append(sample_logits) # add logits for this sample/match to main list

            return np.array(logits) 

    def create_model(self):
        stats = self.stats
        with open("temp2.json", "w") as f:
            f.write(json.dumps(stats, indent=4))

        column_names = []

        for match_stats in stats.values(): # for each match in stats
            for stat_name in match_stats.keys(): # each stat name
                if stat_name not in column_names: # add it to the column names for df if not already in it
                    column_names.append(stat_name)


        df = pd.DataFrame(
        columns=column_names
        )

        for opponent_id in stats: # for each opponent
            for stat_name in stats[opponent_id].keys(): 
                df.loc[opponent_id, stat_name] = stats[opponent_id][stat_name] # put each stat in the df
        
        y_ = np.array(df["Outcome"].values) # Extracting the dependent variable
        y = np.empty((0,3))

        for lis in y_: # reshape and put in the y np array
            lis = np.array(lis).reshape(1, -1)
            y = np.append(y, lis, axis=0)
        


        df = df.drop("Outcome", axis=1) # Removing it from df
        df = df.fillna(0.0)

        X = np.array( # Leaving only the independent variables now
            df
        )

        
        stds = np.std(X, axis=0) # Getting standard deviations of each column
        nonzero_std_mask = stds != 0 # The standard deviations which aren't 0, True in new np array, False if so
    
        
        nonzero_std_mask_ = list(nonzero_std_mask) # Convert to list
        to_drop = [] # Columns to drop in df

        # This needs to be done because when user input is being entered using the dataframe column names, the values need to line up specifically with each coefficient.
        for i in range(0, len(nonzero_std_mask_)):
            if nonzero_std_mask_[i] == False: # If false in here,
                to_drop.append(list(df.columns)[i]) # Draft it to be dropped from df

        df = df.drop(columns=to_drop) # Drop them

        self.columns = list(df.columns)
        X = X[:, nonzero_std_mask] # Keeping columns where standard deviation is not 0 (means there is no dispersion)
        # Also to prevent divide by 0 later on

        std = np.std(X, axis=0) # Now getting the new standard deviation

        mean = np.mean(X, axis=0) # Mean of each column

        self.mean = mean
        self.std = std
        
        X = (X - mean) / std # Scaling the variables

        bias_col = [1 for i in range((X.shape[0]))]

        X_b = np.c_[bias_col, X] # Adding the bias column (intercept)
        X_b = np.nan_to_num(X_b) # Turning any Nan to numbers

            
        betas = np.array([ # for each game get a set of random coefficients 
            [random.random() for i in range(0, len(X_b[0]))],
            [random.random() for i in range(0, len(X_b[0]))],
            [random.random() for i in range(0, len(X_b[0]))]
        ])

        y_hat = self.softmax(self.calc_logits(betas, X_b)) # predicteds


        # for outcome, prediction in zip(y, y_hat):   # loop through each sample pair
        # loss = 0
        # for i in range(len(prediction)):        # loop over classes for that sample
        #     loss += -1 * (outcome[i] * np.log(prediction[i]))
        # total_loss += loss



        eps = 1e-15
        # loss draw win



    
        # loss_derivatives = 1 / y_hat
        # softmax_dervatives = y_hat * (1 - y_hat)

        
        l_rate = 0.1
        iterations = 5000

        for i in range(iterations): # now train the coefficients
            logits = self.calc_logits(betas, X_b)
            y_hat = self.softmax(logits)

            loss = -np.sum(y * np.log(y_hat))

            grad_W = (y_hat - y).T.dot(X_b) / X_b.shape[0]

            betas = betas - l_rate * grad_W
        
        # for i in range(0, len(X)):
        #     for j in range(0, len(column_names)-3):
        #         print(column_names[j], betas[i][j])

        return betas, y, mean, std
    
    def predict_outcome(self, stats):
        coeficients, outcomes, mean, std = self.create_model() # need mean and std to scale inputted data


        sample_data = stats
        bias_col = [1]

        sample_data = (sample_data - mean) / std

        
        sample_data_b = np.c_[bias_col, sample_data] # add bias column/intercept

        final_logits = self.calc_logits(coeficients, sample_data_b)
        probs = self.softmax(final_logits) # get probabilities

        y = probs
     
        y = list(y[0])
        
        index = y.index(max(y))

        return index

    def final_outcome(self, stats): 
        win = 0
        draw = 0
        loss = 0
        outcomes = [loss, draw, win]

        for i in range(1, 2): # repeat prediction 100 times to find highest outputted one
            index = self.predict_outcome(stats)
            outcomes[index] += 1
        
        return outcomes.index(max(outcomes)) # return that
    

# s = np.array([
#     [100 for i in range(0, 43)]
# ])
# x = PredictOutcome(9)
# x.get_data()

# print(x.final_outcome(s))

class PredictPlayerPerformance():
    
    def __init__(self, player_id):
        self.player_id = player_id
        playersinfo_url = f"https://api.sportmonks.com/v3/football/players/{self.player_id}?api_token=p7pnma41hZ54JY3pwMd1GXh3cWykgQYiqUzdVcOlxVcLsvfXblU5B4oTT76M&include=teams"
        self.team_id = requests.get(playersinfo_url).json()["data"]["teams"][0]["team_id"]
        # using more class variables for easier data retrieval in & outside this model
        self.columns = []
        self.fixtures_names = []

        self.beta = None
        self.mean = None
        self.std = None
        self.player_name = None
        self.prev_ratings = None

        self.pred_rating = None
    
    def get_data(self):
        result_ids = [] 

        get_team = f"https://api.sportmonks.com/v3/football/teams/{self.team_id}?api_token={YOUR_TOKEN}&include=latest" # Gets the teams recent results

        recent_results_data = requests.get(get_team).json()

        # with open("results.json", "w") as f:
        #     f.write(json.dumps(recent_results_data, indent=4))

        for i in range(0,2): # getting the id for each result and putting in a list
            result_ids.append(recent_results_data["data"]["latest"][i]["id"])

        ids_str = ",".join(map(str, result_ids)) # Converts each id to a string, then joins them with commas
        results_url = f"https://api.sportmonks.com/v3/football/fixtures/multi/{ids_str}?api_token={YOUR_TOKEN}&include=lineups.details" # Fetches last couple of results

        fixtures_data = requests.get(results_url).json()

        with open("results.json", "w") as f:
            f.write(json.dumps(fixtures_data, indent=4))


        fixtures_stats = {} # The stats of each fixture

        for fixture in fixtures_data["data"]:
            fixture_stats = {} #The stats of one fixture
            self.fixtures_names.append(fixture["name"])

            for player in fixture["lineups"]: # For each player in the lineups
                if player["player_id"] == self.player_id: # Check if its the player we're looking for
                    player_name = player["player_name"] # If it is, get the name
                    for detail in player["details"]: # For all the statistics of the player
                        type_id = detail["type_id"] # get type id
                        finder = Type_finder(id=type_id)
                        type_info = finder.find_type_by_id() # returning the info of the type
                        type_name = type_info["name"] # getting the types name

                        value = detail["data"].get("value", 0) # get the value

                        fixture_stats[type_name] = value # Add to dict for that fixture

            fixtures_stats[fixture["id"]] = fixture_stats 



        with open("temp1.json", "w") as f:
            f.write(json.dumps(fixtures_stats, indent=4))


        column_names = []

        for fixture_values in fixtures_stats.values(): # For each fixtures data
            for stat_name in fixture_values.keys():  # For each stat for each fixture
                if stat_name not in column_names: # If not already added, then add it
                    column_names.append(stat_name)


        df = pd.DataFrame(
            columns=column_names
        ) # New df
        
        for fixture in fixtures_stats:

            df.loc[fixture] = 0.0 # Default everything to 0

            for stat_name in fixtures_stats[fixture]: # Now add in all of the stats which have values for each fixture
                df.loc[fixture, stat_name] = fixtures_stats[fixture][stat_name]
        
        self.player_name = player_name
        return df, player_name

    def create_player_pred_model(self):

        df, player_name = self.get_data() # Gets a dataframe of the data we need

        y = np.array(df["Rating"].values) # Extracting the dependent variable
        

        if "Captain" in df.columns:
            df = df.drop("Captain", axis=1) # Removing it from df
        
    
        df = df.drop("Rating", axis=1) # Removing it from df

        X = np.array( # Leaving only the independent variables now
            df
        )
        
        stds = np.std(X, axis=0) # Getting standard deviations of each column
        nonzero_std_mask = stds != 0 # The standard deviations which aren't 0, True in new np array, False if so
    
        
        nonzero_std_mask_ = list(nonzero_std_mask) # Convert to list
        to_drop = [] # Columns to drop in df

        # This needs to be done because when user input is being entered using the dataframe column names, the values need to line up specifically with each coefficient.
        for i in range(0, len(nonzero_std_mask_)):
            if nonzero_std_mask_[i] == False: # If false in here,
                to_drop.append(list(df.columns)[i]) # Draft it to be dropped from df

        df = df.drop(columns=to_drop) # Drop them

        X = X[:, nonzero_std_mask] # Keeping columns where standard deviation is not 0 (means there is no dispersion)
        # Also to prevent divide by 0 later on

        std = np.std(X, axis=0) # Now getting the new standard deviation

        mean = np.mean(X, axis=0) # Mean of each column
        
        X = (X - mean) / std # Scaling the variables

        bias_col = [1 for i in range((X.shape[0]))]
        self.columns = list(df.columns)

        X_b = np.c_[bias_col, X] # Adding the bias column (intercept)
        X_b = np.nan_to_num(X_b) # Turning any Nan to numbers

        beta = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y) # Normal equation to find coefficients for the linear equation y = x₀b₀ + x₁b₁ .. etc
        beta = beta.reshape(-1, 1) # Reshaping for matrix multiplication
        
    
        dfcolumns_withbias = ["Intercept/bias"] + list(df.columns)

        # for i in range(0, len(dfcolumns_withbias)):
        #     print(dfcolumns_withbias[i], beta[i])
        
        mse, r2 = self.evaluate(y, X_b.dot(beta)) # Getting evaluative variables

    
        self.beta, self.mean, self.std, self.player_name, self.prev_ratings = beta, mean, std, player_name, y
        # beta -> coefficients -> Essesntially what we need for the model to now just input X values to get y rating
        # y is the models that the ratings were trained on, gonna be used later on a bar chart and also as extra info

    def predict_peformance(self, stats):
        
        stats = (stats - self.mean) / self.std # Scaling inputted data

    

        stats_b = np.concatenate([[1.0], stats])



        # Then clipping it to be between 0 and 10 if it is outside that range
        self.pred_rating = np.clip(stats_b.dot(self.beta), 0.0, 10.0) 
        self.pred_rating = np.trunc(self.pred_rating * 100) / 100 # Rouding to 2 decimal places        

    def evaluate(self,y, y_pred):

        mse = np.mean((y - y_pred) ** 2) #Mean squared error
        rss = np.sum((y - y_pred) ** 2) # Residual Sum of Squares
        tss = np.sum(( y - np.mean(y) ) ** 2) # Total Sum of Squares

        r2 = 1 - (rss/tss) # Coefficient of determination (R²)

        return mse, r2

    def bar_chart(self, req):
        
        get_team_fixtures = requests.get(f"https://api.sportmonks.com/v3/football/teams/{self.team_id}?api_token={YOUR_TOKEN}&include=upcoming").json()

        # with open("results.json", "w") as f:
        #     f.write(json.dumps(get_team_fixtures, indent=4))

        team_name = get_team_fixtures["data"]["name"]

        next_fixture_name = get_team_fixtures["data"]["upcoming"][0]["name"] # Getting the next fixtures name
    
        fixtures = self.fixtures_names
        fixtures.append(next_fixture_name) # Adding to the list of fixtures, same with ratings
        self.prev_ratings = np.append(self.prev_ratings, self.pred_rating)
        
        

        plt.figure(figsize=(len(fixtures), 3)) # Plot should be as big as the number of fixtures involved

        for i in range (0, len(fixtures)): # For range in len of fixtures and ratings

            fixture_name = fixtures[i] # Get the fixture name

            #Strip it down to just the oppositions name
            fixture_name = fixture_name.removesuffix(f"vs {team_name}").removeprefix(f"{team_name} vs").strip()

            
            # Add on the vs
            fixture_name = " vs " + fixture_name

            # But if it's the last one (the one to be predicted)
            if i == len(fixtures) - 1:
                fixture_name = fixture_name.replace("vs", " ").strip() # Take out the vs 
                fixture_name = f"Predicted rating vs {fixture_name}" # Put into this string for the label


            plt.bar(x=fixture_name, height=self.prev_ratings[i], width=0.5, color=self.pick_color(self.prev_ratings[i]))
            plt.text(x=fixture_name, y=self.prev_ratings[i]+0.1, s=str(self.prev_ratings[i]), ha='center')
            plt.ylim(0,10)
            plt.yticks(np.arange(0, 10.5, 0.5)) 
            # Plot bar chart
        
            plt.title(f"Ratings for {self.player_name} ({team_name})", pad=20) # Title pf plot, pad -> padding

            time_img = str(round(time.time())) # get current time that analysis took place
            filename = f"{req.session.get("user_id")}-{time_img}" # create a filename

            image_path = f"Main/static/images/analysis/pred_performance/{filename}.png" # path to write image to in the project

            figure = plt.gcf()  # get current figure
            figure.set_size_inches(12, 9) # set figure's size manually to your full screen (32x18)
            plt.savefig(image_path, bbox_inches='tight') # bbox_inches removes extra white spaces

            django_path = f"images/analysis/pred_performance/{filename}.png" # This will be the path that django template will use to get the static image for html 
        
        fixtures.pop()
        self.prev_ratings = np.delete(self.prev_ratings, -1)
        
        return django_path

    # hsv red default (0, 100, value)
    def pick_color(self, rating):

        if rating >= 7: # So green ratings

            if rating == 10: # If it's 10 give preselected color
                return "#00380f"

            # for greens, we want a deeper green to represent a higher rating

            # Else use this formula if its not 10, to calculate the brightness of the green
            value = 1 - rating/10 
            value /= 0.3

            if value > 1:
                value = 0.9

            color = hsv_to_rgb([0.33, 1, value])


        elif rating > 4: # For oranges, a deeper orange means a lower rating
            # Use this calculation
            value = 0.75*rating
            value = ( ( ( rating - 1) * 0.75 ) / 10 ) + 0.55
            color = hsv_to_rgb([0.111, 1.0, value])
        else: 
            # And this one for reds, a deeper red means a lower rating
            value = 0.25*rating
            color = hsv_to_rgb([0, 1, value])
        
        return color