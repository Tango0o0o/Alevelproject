from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.template import loader
from .forms import SignUpForm


def statball(req):
    return render(req, "default.html", {"name" : "John"})


# Here, validation and user creation takes place
def signup(req):
    form = SignUpForm()
    email_msg = ""
    password_msg = ""

    # Post meaning data is being sent to the server; i.e. from the user
    if req.method == "POST":
        form = SignUpForm(req.POST)
        
        
        valid_email = form.is_valid_email()

        if valid_email != True:
            email_msg = valid_email
        
        valid_pass = form.is_valid_password()

        if valid_pass != True:
            password_msg = valid_pass



    return render(req, "accounts/signup.html", {"form":form, "email_msg":email_msg, "password_msg":password_msg})