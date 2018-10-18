from django.urls import path,include
from . import views
from django.contrib.auth import views as auth_views
from .views import TimeSeriesDailyAdjusted,CompanyListView,CompanyData
from django.conf.urls import url,include
from rest_framework import routers

router = routers.DefaultRouter()
router.register(r'data', views.CompanyData)

urlpatterns = [
    url(r'^filterData/', include(router.urls)),
    path('', views.index, name='index'),
    path('register/', views.register, name='register'),
    path(
        'change-password/',
        auth_views.PasswordChangeView.as_view(template_name='change-password.html'),
    ),
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),
    path('search/', views.get_name, name='get_name'),
    path('search/', views.search, name='search'),
    path('api/chart/data/', TimeSeriesDailyAdjusted.as_view()),
    path('api/filter/', CompanyListView.as_view()),
]

urlpatterns += [
    path('api-auth/', include('rest_framework.urls')),
]