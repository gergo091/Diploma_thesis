# -*- coding: utf-8 -*-
from django.shortcuts import render_to_response, render, redirect
from django.template import RequestContext
from django.http import HttpResponseRedirect, HttpResponseBadRequest
from django.core.urlresolvers import reverse
import subprocess, time, os, zipfile, shutil

from .models import Document, neural_network_param, Task
from .forms import DocumentForm

def index(request):
    return HttpResponseRedirect("list.html")

def get_page_objects():
    documents = Document.objects.all()
    param = neural_network_param.objects.all()
    form = DocumentForm()
    tasks = Task.objects.all()

    return documents, param, form, tasks


def list(request):
    # Handle file upload
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            newdoc = Document(docfile=request.FILES['docfile'])
            newdoc.save()

            # Redirect to the document list after POST
            #return HttpResponseRedirect(reverse('app.views.list'))

    documents, param, form, tasks = get_page_objects()

    return render_to_response(
                'list.html',
                {'documents': documents, 'form': form, 'tasks': tasks, 'param': param},
                context_instance=RequestContext(request)
            )


def add_params(request):
    return render_to_response('add_params.html', context_instance=RequestContext(request))

def param(request):
    if request.method == 'POST':

        # Preprocessing & NN parameters
        nn = neural_network_param(name=request.POST.get('name'),
                                  description=request.POST.get('desc'),
                                  size_of_block_orientation=request.POST.get('size_of_block_orientation'),
                                  size_of_block_gabor=request.POST.get('size_of_block_gabor'),
                                  sigma=request.POST.get('sigma'),
                                  var_lambda=request.POST.get('var_lambda'),
                                  gamma=request.POST.get('gamma'),
                                  input_basic=request.POST.get('input_basic'),
                                  output_basic=request.POST.get('output_basic'),
                                  layers_basic=request.POST.get('layers_basic'),
                                  neurons_hidden1=request.POST.get('neurons_hidden1'),
                                  input_complex=request.POST.get('input_complex'),
                                  output_complex=request.POST.get('output_complex'),
                                  layers_complex=request.POST.get('layers_complex'),
                                  neurons_hidden2=request.POST.get('neurons_hidden2'),
                                  desired_error=request.POST.get('desired_error'),
                                  max_epochs=request.POST.get('max_epochs'),
                                  fann_set_activation_function_hidden=request.POST.get(
                                      'fann_set_activation_function_hidden'),
                                  fann_set_activation_function_output=request.POST.get(
                                      'fann_set_activation_function_output'),
                                  )
        nn.save()

        documents, param, form, tasks = get_page_objects()

        return render_to_response(
            'list.html',
            {'documents': documents, 'param': param, 'form': form, 'tasks': tasks, 'alert': ' Algorithm parameters were saved!'},
            context_instance=RequestContext(request)
        )


def detail(request, pk):
    nn = neural_network_param.objects.get(pk=pk)
    return render_to_response('detail.html', {
        'param': nn}, context_instance=RequestContext(request))

def detail_delete(request, pk):
    nn = neural_network_param.objects.get(pk=pk)
    nn.delete()
    form = DocumentForm()  # A empty, unbound form


    # Load documents for the list page
    documents, param, form, tasks = get_page_objects()

    alert = 'Algorithm parameters were deleted!'

    return render_to_response(
        'list.html',
        {'documents': documents, 'form': form, 'param': param, 'tasks': tasks, 'alert': alert},
        context_instance=RequestContext(request)
    )

def runtask(request):
    data = request.POST
    print data

    if 'document' not in data:
        return HttpResponseBadRequest('Missing parameter document')

    if 'algorithm' not in data:
        return HttpResponseBadRequest('Missing parameter algorighm')

    document = Document.objects.get(pk=data['document'])
    algorithm = neural_network_param.objects.get(pk=data['algorithm'])
    date = time.strftime("%Y-%m-%d")
    train = data.get('train')

    print document
    print algorithm

    if(train):
        print 'true'


    task = Task(status=Task.RUNNING, image=document, task_params=algorithm, date=date)
    task.save()

    documents, param, form, tasks = get_page_objects()

    alert = ' Starting task for image: ' + str(document) + '  with algoritm parameters: '+ str(algorithm.name)

    #run subprocess - C program for fingerprint processing with arguments
    if (train):
        subprocess.Popen(['/home/gregor/projects/dp_image_processing/dp_iamge_processing/media/runner.sh', str(task.id),
                          '0', str(document), str(algorithm.input_basic), str(algorithm.output_basic),
                          str(algorithm.layers_basic), str(algorithm.input_complex), str(algorithm.output_complex),
                          str(algorithm.layers_complex), str(algorithm.desired_error), str(algorithm.max_epochs),
                          str(algorithm.neurons_hidden1), str(algorithm.neurons_hidden2), str(algorithm.fann_set_activation_function_hidden),
                          str(algorithm.fann_set_activation_function_output), str(algorithm.size_of_block_orientation),
                          str(algorithm.size_of_block_gabor), str(algorithm.sigma), str(algorithm.var_lambda),
                          str(algorithm.gamma)], bufsize=0)
    else:
        subprocess.Popen(
            ['/home/gregor/projects/dp_image_processing/dp_iamge_processing/media/runner.sh', str(task.id), '1',
             str(document), str(algorithm.input_basic), str(algorithm.output_basic), str(algorithm.layers_basic),
             str(algorithm.input_complex), str(algorithm.output_complex), str(algorithm.layers_complex),
             str(algorithm.desired_error), str(algorithm.max_epochs), str(algorithm.neurons_hidden1),
             str(algorithm.neurons_hidden2), str(algorithm.fann_set_activation_function_hidden),
             str(algorithm.fann_set_activation_function_output), str(algorithm.size_of_block_orientation),
             str(algorithm.size_of_block_gabor), str(algorithm.sigma), str(algorithm.var_lambda),
             str(algorithm.gamma)], bufsize=0)

    return HttpResponseRedirect("/")
    #return render(request, 'list.html',
    #                  {'documents': documents, 'form': form, 'param': param, 'tasks': tasks, 'alert': alert},context_instance=RequestContext(request))



#def output(request, pk):
 #   task = Task.objects.get(pk=pk)
  #  return render_to_response('output.html', {
   #     'task': task}, context_instance=RequestContext(request))

def task_delete(request, pk):
    task = Task.objects.get(pk=pk)
    task.delete()

    # Load documents for the list page
    documents, param, form, tasks = get_page_objects()

    alert = ' Task were deleted!'

    return HttpResponseRedirect("/")
    #return render(request, 'list.html',
     #             {'documents': documents, 'form': form, 'param': param, 'tasks': tasks, 'alert': alert})

def image_delete(request):
    data = request.POST
    document = Document.objects.get(pk=data['image'])
    print document
    document.delete()

    # Load documents for the list page
    documents, param, form, tasks = get_page_objects()

    alert = ' Image were deleted!'

    return HttpResponseRedirect("/")
    #return render(request,'list.html', {'documents': documents, 'form': form, 'param': param, 'tasks': tasks, 'alert': alert})
    #

def output(request, pk):
    task = Task.objects.get(pk=pk)
    path='/home/gregor/projects/dp_image_processing/dp_iamge_processing/media/tasks/' +str(task.id)+ '/output/'
    # insert the path to your directory
    img_list =os.listdir(path)
    img_list.sort()
    count = len(img_list)
    #print count
    path_dir = '/home/gregor/projects/dp_image_processing/dp_iamge_processing/media/tasks/' +str(task.id) +'/output'

    #create ZIP file for download
    shutil.make_archive("outputs", "zip", str(path_dir))

    #move ZIP file to /media
    subprocess.Popen(['/home/gregor/projects/dp_image_processing/dp_iamge_processing/media/movezip.sh'], bufsize=0)

    if(count > 2):
        return render_to_response('output.html', {'images_all': img_list, 'task': task}, context_instance=RequestContext(request))
    elif (count < 10):
        return render_to_response('output.html', {'images_some': img_list, 'task': task},
                                  context_instance=RequestContext(request))
    else:
        return render_to_response('output.html', {'images_other': img_list, 'task': task},
                                  context_instance=RequestContext(request))

