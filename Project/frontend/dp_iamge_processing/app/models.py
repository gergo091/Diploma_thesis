# -*- coding: utf-8 -*-
import datetime

from django.db import models
from django.db.models.signals import post_init
from django.dispatch import receiver
from django.utils import timezone

class Document(models.Model):
    docfile = models.FileField(upload_to='documents/')
    pub_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.docfile.name


class neural_network_param(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField(max_length=600)

    FANN_LINEAR = 0
    FANN_THRESHOLD = 1
    FANN_THRESHOLD_SYMMETRIC = 2
    FANN_SIGMOID = 3
    FANN_SIGMOID_STEPWISE = 4
    FANN_SIGMOID_SYMMETRIC = 5
    FANN_SIGMOID_SYMMETRIC_STEPWISE = 6
    FANN_GAUSSIAN = 7
    FANN_GAUSSIAN_SYMMETRIC = 8
    FANN_GAUSSIAN_STEPWISE = 9
    FANN_ELLIOT = 10
    FANN_ELLIOT_SYMMETRIC = 11
    FANN_LINEAR_PIECE = 12
    FANN_LINEAR_PIECE_SYMMETRIC = 13
    FANN_SIN_SYMMETRIC = 14
    FANN_COS_SYMMETRIC = 15
    FANN_SIN = 16
    FANN_COS = 17
    FANN_ACTIVATIONFUNC_CHOICES = (
        (FANN_LINEAR, 'FANN_LINEAR'),
        (FANN_THRESHOLD, 'FANN_THRESHOLD'),
        (FANN_THRESHOLD_SYMMETRIC, 'FANN_THRESHOLD_SYMMETRIC'),
        (FANN_SIGMOID, 'FANN_SIGMOID'),
        (FANN_SIGMOID_STEPWISE, 'FANN_SIGMOID_STEPWISE'),
        (FANN_SIGMOID_SYMMETRIC, 'FANN_SIGMOID_SYMMETRIC'),
        (FANN_SIGMOID_SYMMETRIC_STEPWISE, 'FANN_SIGMOID_SYMMETRIC_STEPWISE'),
        (FANN_GAUSSIAN, 'FANN_GAUSSIAN'),
        (FANN_GAUSSIAN_SYMMETRIC, 'FANN_GAUSSIAN_SYMMETRIC'),
        (FANN_GAUSSIAN_STEPWISE, 'FANN_GAUSSIAN_STEPWISE'),
        (FANN_ELLIOT, 'FANN_ELLIOT'),
        (FANN_ELLIOT_SYMMETRIC, 'FANN_ELLIOT_SYMMETRIC'),
        (FANN_LINEAR_PIECE, 'FANN_LINEAR_PIECE'),
        (FANN_LINEAR_PIECE_SYMMETRIC, 'FANN_LINEAR_PIECE_SYMMETRIC'),
        (FANN_SIN_SYMMETRIC, 'FANN_SIN_SYMMETRIC'),
        (FANN_COS_SYMMETRIC, 'FANN_COS_SYMMETRIC'),
        (FANN_SIN, 'FANN_SIN'),
        (FANN_COS, 'FANN_COS')
    )

    input_basic = models.IntegerField(null=True)
    output_basic = models.IntegerField(null=True)
    layers_basic = models.IntegerField(null=True)

    input_complex = models.IntegerField(null=True)
    output_complex = models.IntegerField(null=True)
    layers_complex = models.IntegerField(null=True)

    desired_error = models.FloatField(null=True)
    max_epochs = models.IntegerField(null=True)
    #epochs_between_reports = models.IntegerField(null=True)
    neurons_hidden1 = models.IntegerField(null=True)
    neurons_hidden2 = models.IntegerField(null=True)

    fann_set_activation_function_hidden = models.IntegerField(choices=FANN_ACTIVATIONFUNC_CHOICES, null=True)
    fann_set_activation_function_output = models.IntegerField(choices=FANN_ACTIVATIONFUNC_CHOICES, null=True)

    size_of_block_orientation = models.IntegerField(null=True)
    size_of_block_gabor = models.IntegerField(null=True)
    sigma = models.IntegerField(null=True)
    var_lambda = models.FloatField(null=True)
    gamma = models.FloatField(null=True)

    def __str__(self):
        return self.name



class Task(models.Model):
    RUNNING = 1
    COMPLETED = 2
    STATUS_CHOICES = (
        (RUNNING, 'Running'),
        (COMPLETED, 'Completed'),
    )
    status = models.IntegerField(choices=STATUS_CHOICES, default=RUNNING)
    image = models.ForeignKey(Document)
    task_params = models.ForeignKey(neural_network_param)
    date = models.DateField()




@receiver(post_init, sender=Task)
def get_task_status(sender, instance, **kwargs):
    if instance.id:
        try:
            file = open('/home/gregor/projects/dp_image_processing/dp_iamge_processing/media/tasks/' + str(instance.id) + '/status')

            status = file.readline()

            if status.strip() == 'COMPLETED':
                print 'saving'
                instance.status = Task.COMPLETED
                instance.save()

            print "status of the task is " + status
        except:
            pass
