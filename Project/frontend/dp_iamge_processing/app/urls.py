# -*- coding: utf-8 -*-
from django.conf.urls import patterns, url

urlpatterns = patterns('app.views',
    url(r'^$', 'list', name='list'),
    url(r'^list/$', 'list', name='list'),
    url(r'^param/$','param', name='param'),
    url(r'^detail/(?P<pk>\d+)/$', 'detail', name='detail'),
    url(r'^detail_delete/(?P<pk>\d+)/$', 'detail_delete', name='detail_delete'),
    url(r'^add_params/$', 'add_params', name="add_params"),
    url(r'^runtask/$', 'runtask', name='runtask'),
    url(r'^output/(?P<pk>\d+)/$', 'output', name='output'),
    url(r'^task_delete/(?P<pk>\d+)/$', 'task_delete', name='task_delete'),
    url(r'^image_delete/$', 'image_delete', name='image_delete'),
)