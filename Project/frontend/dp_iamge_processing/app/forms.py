# -*- coding: utf-8 -*-
from django import forms
from .models import neural_network_param

class DocumentForm(forms.Form):
    docfile = forms.FileField(
        label='Select a file',
        help_text=' only image(.bmp, .jpg,...)'
    )