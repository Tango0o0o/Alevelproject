from django.contrib import admin
from .models import User, Analysis
# Register your models here.

# This adds the model to the admin panel so I can see the objects created from them
admin.site.register(User)
admin.site.register(Analysis)