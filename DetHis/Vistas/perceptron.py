from django.views import View
from django.http import JsonResponse
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt

from DetHis.Helpers.npEncoder import NumpyEncoder
from ..Helpers.audioHelpers import *
from ..Helpers.loadModel import *
import json


class Perceptron(View):

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
            deleteFile(fileName)
            model = getPerceptron()
            predict = model.predict(mfcc, batch_size=1)
            fullPrediction = json.dumps(predict, cls=NumpyEncoder)
            best_index = np.argmax(predict)
            acento = getAcentos()[best_index]
            return JsonResponse({'status': 'ok', 'metodo': 'post', 'mfcc Shape': mfcc.shape, 'predictionIndex': int(best_index), 'acento': acento, 'prediction': fullPrediction})
        except Exception as e:
            deleteFile(str(request.FILES['wav']))
            return JsonResponse({'status': 'error', 'error': str(e)})

    def put(self, request):
        return JsonResponse({'status': 'ok'})

    def delete(self, request):
        return JsonResponse({'status': 'ok'})
