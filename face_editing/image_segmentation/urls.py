from django.urls import path

from . import views


urlpatterns = [
    # path('', views.index, name='index'),
    path('', views.segmentation_view, name='seg_view'),
    path('gen', views.generation_view, name='gen_view')
]