from django.views import View
from django.http import JsonResponse
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from ..Helpers.audioHelpers import *
from ..Helpers.loadModel import *


class KNN(View):

    @method_decorator(csrf_exempt)
    def dispatch(self, request, *args, **kwargs):
        return super().dispatch(request, *args, **kwargs)

    def get(self, request):
        return JsonResponse({'status': 'ok', 'metodo': 'get', 'request': request.GET})

    def post(self, request):
        try:
            fileName = request.FILES['wav']
            fileName = str(fileName)
            path = saveFile(fileName,
                            request.FILES.get('wav'))
            mfcc = generateMFCCArray(path=path)
            mfcc = mfcc.reshape(mfcc.shape[0], -1)
            deleteFile(request.FILES.get('wav').name)
            model = getKNN()
            predict = model.predict(mfcc)
            print(int(predict[0]))
            predictedIndex = int(predict[0])
            acento = getAcentos()[predictedIndex]
            return JsonResponse({'status': 'ok', 'metodo': 'post', 'mfcc Shape': mfcc.shape, 'predictionIndex': int(predictedIndex), 'acento': acento})
        except Exception as e:
            deleteFile(request.FILES.get('wav').name)
            return JsonResponse({'status': 'error', 'error': str(e)})

    def put(self, request):
        return JsonResponse({'status': 'ok'})

    def delete(self, request):
        return JsonResponse({'status': 'ok'})
