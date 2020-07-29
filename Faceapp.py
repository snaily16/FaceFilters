from VideoStream import VideoStream
from FaceFilters import FaceFilters
from faceTk import GUIFace
import time

filters = ['glasses.png', 'sunglasses.png', 'sunglasses1.png', 'sunglasses2.png', \
        'dog.png', 'rabbit.png','moustache.png', 'moustache1.png', 'ironman.png', 'capAmerica.png']

vs = VideoStream(0).start()
fc = FaceFilters(filters)
time.sleep(2.0)
gui = GUIFace(vs,fc,'output')
