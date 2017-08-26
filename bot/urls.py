from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^submit', views.submit, name='submit'),
]

# a dummy method for making db operations newboston video tutorial
#url(r'^(?P<message_id>[0-9]+)/$', views.details, name='details'),
