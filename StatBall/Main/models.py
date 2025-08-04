from django.db import models

# Create your models here.

class User(models.Model):
    email = models.CharField(name="email", max_length=99)
    password = models.CharField(name="password")
    # ID automatically created

    def __str__(self):
        return self.email