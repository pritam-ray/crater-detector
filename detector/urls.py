from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('process/<int:image_id>/', views.process_image, name='process_image'),
    path('result/<int:image_id>/', views.result, name='result'),
    path('gallery/', views.gallery, name='gallery'),
]
