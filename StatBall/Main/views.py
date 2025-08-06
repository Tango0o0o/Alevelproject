from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.template import loader
from .forms import SignUpForm, LoginForm
from .models import User
import bcrypt


# Extra functions

# Checks if there is a user logged in
def is_logged_in(req):
    if req.session.get("user_id"): # Checks if a user id exists in the session
        return True
    return False


# The actual views that lead to pages

def statball(req):
    print(is_logged_in(req))
    return render(req, "default.html", {"name" : f"{str(is_logged_in(req))}"})

# Here, validation and user creation takes place
def signup(req):

    if is_logged_in(req): # if logged in, you can't sign up
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

    
    return render(req, "accounts/signup.html", {"form":form, "email_msg":email_msg, "password_msg":password_msg})

# Authenticates the user and creates a new session
def login(req):

    if is_logged_in(req):
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

    return render(req, "accounts/login.html", {"login_form": login_form, "message":message}) # returning the webpage


# Logs the user out by clearing the session
def logout(req):
    
    if is_logged_in(req): # Only execute the process of the users is actually logged in
        req.session.flush()
    
    return redirect("home")