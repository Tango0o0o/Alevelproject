from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader

def statball(req):
    return render(req, "default.html", {"name" : "John"})