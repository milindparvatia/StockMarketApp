from django.urls import path
from . import views

from .views import TimeSeriesDailyAdjusted,MACD
# Tweeter

urlpatterns = [
    path('', views.index, name='index'),
    path('register/', views.register, name='register'),
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),
    path('search/', views.get_name, name='get_name'),
    path('search/', views.search, name='search'),
    path('api/chart/TimeSeriesDailyAdjusted/', TimeSeriesDailyAdjusted.as_view()),
    path('api/chart/MACD/', MACD.as_view()),
    # path('api/chart/Tweeter/', Tweeter.as_view()),
]