from django.contrib import admin
from .models import Document, neural_network_param, Task


admin.site.register(Document)
admin.site.register(neural_network_param)
admin.site.register(Task)
# Register your models here.
