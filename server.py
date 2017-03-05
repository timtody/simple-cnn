from flask import Flask, render_template
from scipy.misc import toimage
import numpy as np
import os
import model

app = Flask(__name__)

@app.route('/')
def hello_world():

    model.model.fromParams('nets/tmpk3owhr64')
    pictures = []
    piclist = os.listdir('static')
    for x in range(30):
        rng = np.random.randint(0, 100)
        dic = {}
        dic['path'] = piclist[rng]
        dic['ref'] = rng
        pictures.append(dic)
    print(pictures)
    th = ['Name', 'Batch size']
    models = [f.split(".") for f in os.listdir('nets')]
    return render_template('index.html', models=models, th=th, pictures=pictures)

@app.route('/<toPredict>')
def prediction(toPredict):

    model.model.fromParams('nets/tmpk3owhr64')
    prediction = model.model.predict(model.X[int(toPredict)])
    pictures = []
    piclist = os.listdir('static')
    for x in range(30):
        rng = np.random.randint(0, 100)
        dic = {}
        dic['path'] = piclist[rng]
        dic['ref'] = piclist[rng].split("_")[0]
        pictures.append(dic)
    print(pictures)
    th = ['Name', 'Batch size']
    models = [f.split(".") for f in os.listdir('nets')]
    return render_template('index_p.html', models=models, th=th, pictures=pictures, prediction=prediction, target=int(model.y[int(toPredict)]))





# for x in range(81):
#     rng = np.random.randint(0, 70000)
#     datum = model.X[rng]
#     datum.shape = (28, 28)
#     im = toimage(datum)
#     #im.show()
#     im.save('img/' + str(rng) + '_' + str(model.y[rng]) + ".bmp")
#


app.run()