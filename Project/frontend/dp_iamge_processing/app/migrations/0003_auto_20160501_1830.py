# -*- coding: utf-8 -*-
# Generated by Django 1.9.5 on 2016-05-01 16:30
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0002_auto_20160428_2231'),
    ]

    operations = [
        migrations.AlterField(
            model_name='neural_network_param',
            name='gamma',
            field=models.FloatField(null=True),
        ),
        migrations.AlterField(
            model_name='neural_network_param',
            name='var_lambda',
            field=models.FloatField(null=True),
        ),
    ]
