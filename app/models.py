# Create your models here.
from django.db import models

class CompanyList(models.Model):
    company_name = models.CharField(max_length=20)

    def __str__(self):
        return self.company_name