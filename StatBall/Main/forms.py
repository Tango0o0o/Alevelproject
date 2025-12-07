from django import forms # using the built in django forms class
from .models import User
import bcrypt
import requests as req
YOUR_TOKEN = "p7pnma41hZ54JY3pwMd1GXh3cWykgQYiqUzdVcOlxVcLsvfXblU5B4oTT76M"

class SignUpForm(forms.Form):
    
    email = forms.CharField(max_length=100, 
        label="",
                            
        widget=forms.TextInput(
        attrs={
            "placeholder" : "Email",
            "class" : "form-control"
        }
    ))

    password = forms.CharField(
        label="",
        widget=forms.PasswordInput(
            attrs={
                "placeholder" : "Password",
                "class" : "form-control form-input"
                }
            )
    )

    # This validates the email and field before allowing submission
    def is_valid_email(self):

        valid_tld = self.is_TLD()
        if valid_tld != True:
            return valid_tld
        
        # Checks if there is one "@" symbol
        if self.get_email().count("@") != 1:
            return "Email must contain one @"
        
        valid_domain = self.check_domain()
        if valid_domain != True:
            return valid_domain
        
        valid_local = self.check_local()
        if valid_local != True:
            return valid_local
        
        if len(self.get_email()) >= 100:
            return "Email must be less than 100 characters"
        
        duplicate = self.duplicate_email()
        if duplicate != False:
            return duplicate
        
        return True
    
    # Checks if a password is valid based on criteria
    def is_valid_password(self):
        password = self.get_password().strip()
        
        if len(password) < 8:
            return "Password must be at least 8 characters"

        #Check for uppercase
        uppercase = False
        for letter in password:
            if letter.isupper():
                uppercase = True
                break
        
        if uppercase == False:
            return "Password must contain at least one uppercase letter"
        
        # Check for number 
        digit = False
        for letter in password:
            if letter.isdigit():
                digit = True
                break
        
        if digit == False:
            return "Password must contain at least one number"
        
        # Checks if each char is not a letter or number; hence special char
        alphanum = True 
        for letter in password:
            if not letter.isalnum():
                alphanum = False
                break
        
        if alphanum == True:
            return "Password must contain at least one special character"
        
        return True

    # Checks if there is a valid TLD
    def is_TLD(self):
 
        if self.get_email()[-4:] != ".com" and self.get_email()[-3:] != ".uk":
            return "Email must be UK valid ending in .uk or .com"

        return True
    
    # Checks for a domain after the @ sign
    def check_domain(self):
        at_index = self.get_email().find("@")
        tld_index = self.get_email().find(".com")
        
        # Will be either .com or .uk, if finding .com return -1, then .uk must be in there
        if tld_index == -1:
            tld_index = self.get_email().find(".uk")
        
        # If the TLD and @ are next to each other, meaning nothing is inbetween, there is no domain
        if tld_index - at_index == 1: 
            return "Email must contain a domain after @"
        
        return True
    
    # Checks if there is a local domain before the "@" sign
    def check_local(self):
        if self.get_email().find("@") == 0:
            return "Email must contain a local domain before the '@' sign"
        return True

    # Checks if this is email already in the system
    def duplicate_email(self):

        # This filters the user objects by the stated value 'email'
        # .exists() returs a boolean if there is anything in the queryset from .filter()
        if User.objects.filter(email=self.get_email()).exists():
            return "An account with this email already exists"
        else:
            return False
    
    # Returns email in this form
    def get_email(self):
        return self.data.get("email")
    
    # Returns plaintext password in this form
    def get_password(self):
        return self.data.get("password")

class LoginForm(forms.Form):

    email = forms.CharField(max_length=100, 
        label="",
                            
        widget=forms.TextInput(
        attrs={
            "placeholder" : "Email",
            "class" : "form-control"
        }
    ))

    password = forms.CharField(
        label="",
        widget=forms.PasswordInput(
            attrs={
                "placeholder" : "Password",
                "class" : "form-control form-input"
                }
            )
    )
    
    # This checks if an object with the entered email exists in the system. If so, then it checks if the password associated with the account matches with the one entered.
    def authenticate(self):
        user = User.objects.filter(email=self.get_email()) # Returs a query set

        if user.exists(): # Checks of there is anything in the query set

            user = user[0] # gets the user object in the queryset
      
            entered_password_bytes = self.get_password().encode() 
            account_password = user.password[2:-1].encode() # removing the bytes literal from the password in the DB

            valid_password = bcrypt.checkpw(entered_password_bytes, account_password) # return True if they match
        
            return [valid_password, user]

        return [False]
        
    # Returns email in this form
    def get_email(self):
        return self.data.get("email")
    
    # Returns plaintext password in this form
    def get_password(self):
        return self.data.get("password")
    
class SimilarPlayersForm(forms.Form):


    # Reused from email + password
    player_name = forms.CharField(max_length=60,
        label="",
        widget=forms.TextInput(
            attrs={
                "placeholder" : "Enter players full name",
                "class" : "form-control"
            }
    ))

    team_name = forms.CharField(max_length=60,
        label="",
        widget=forms.TextInput(
            attrs={
                "placeholder" : "Enter teams full name",
                "class" : "form-control"
            }
    ))

    # Returns email in this form
    def get_team_id(self):
        requested_team = self.data.get("team_name")
        
        # Because the user will input a string, we will find a list of the closest matches to the input, and select the top one
        get_teams__url = f"https://api.sportmonks.com/v3/football/teams/search/{requested_team}?api_token={YOUR_TOKEN}"
        data = req.get(get_teams__url).json()

        if "message" in data: # If theres a message, theres an error
            return False

        team_id = data["data"][0]["id"]

        return team_id

    # Returns plaintext player name in this form
    def get_player_id(self):
        requested_player = self.data.get("player_name")
        # Because the user will input a string, we will find a list of the closest matches to the input, and select the top one
        get_players_url = f"https://api.sportmonks.com/v3/football/players/search/{requested_player}?api_token={YOUR_TOKEN}"

        data = req.get(get_players_url).json()
        
        if "message" in data: # If theres a message, theres an error
            return False
        
        player_id = data["data"][0]["id"]

        return player_id
    
class PredictPlayerPerformanceForm(SimilarPlayersForm): # Just inherit stuff from the Similar players form
    
    team_name = None # except we don't need a team name

class PredictMatchOutcomeForm(SimilarPlayersForm): # Just inherit stuff from the Similar players form
    
    player_name = None # except we don't need a player name
    