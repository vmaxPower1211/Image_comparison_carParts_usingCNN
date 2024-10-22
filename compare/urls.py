# urls.py (in the app folder)

from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
urlpatterns = [
    path('', views.compare_images, name='compare_images'),
]