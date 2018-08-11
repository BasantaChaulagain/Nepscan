# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import os

from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from modules.predict import predictions


def index(request):
    return render(request, 'main/index.html')

@api_view(["GET"])
def nth(request):
    return Response("response")

@api_view(["POST"])
def classify(request):
    try:
        os.system('rm -rf ./resources/predict/here/*')
        myfile = request.FILES['image']
        fs = FileSystemStorage()
        fs.save('./resources/predict/here/'+myfile.name, myfile)
        index = predictions()['classes'][0]
        prob = predictions()['probabilities'][0][index]
        # return Response({'index':str(index), 'prob':str(prob)})
        return Response([str(index), str(prob)])
    except ValueError as e:
        print e.args
        return Response(e.args[0], status.HTTP_400_BAD_REQUEST)

