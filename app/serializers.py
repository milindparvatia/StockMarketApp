from rest_framework import serializers
from app.models import CompanyList

class CompanyListSerializer(serializers.ModelSerializer):
    class Meta:
        model = CompanyList
        fields = ('company_name',)