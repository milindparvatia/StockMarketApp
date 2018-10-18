from rest_framework import serializers
from app.models import CompanyList

class CompanyListSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = CompanyList
        fields = ('company_name',)