from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.template import loader
from django.contrib.auth.forms import UserCreationForm

def statball(req):
    return render(req, "default.html", {"name" : "John"})

# 