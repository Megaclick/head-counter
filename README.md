# head-counter

## Introduccion 

Este repositorio contiene el trabajo hecho para el ramo de Vision por Computador [ELO-XXX] de la universidad federico santamaria.

El objetivo de este repositorio es entregar una herramienta para hacer seguimiento de personas en el conexto de la evasión dentro de los buses de transantiago.

### Etapas

El pipeline consta de 3 etapas, deteccion, seguimiento y conteo.

##### Deteccion

Para realizar la deteccion se utilizo [Darknet](https://github.com/AlexeyAB/darknet), en donde se entrenó la version de yolov4-tiny en base al dataset entregado por [Scut-Head](https://github.com/HCIILAB/SCUT-HEAD-Dataset-Release).

Tambien se utilizaron imagenes de los mismos recorridos de los buses, las cuales se pueden ver en la carpeta imagenes (estas imagenes es solo una pequeña muestra del dataset completo). 

Se utilizo data argumentation para mejorar la variabilidad dentro del dataset de buses, para ello se utilizo la herramienta [imgaug](https://github.com/aleju/imgaug)

Para la metrica de evaluacion, se utilizo la misma herramienta del calculo de mAP de  [Darknet](https://github.com/AlexeyAB/darknet), en donde los resultados obtenidos se muestran en la siguiente tabla.

Estos resultados fueron generados en base al set de testing de Scut-Head-B dada la similiritud de las imagenes con el contexto de buses.

Para la transformacion a TensorRT se utilizo como base [tensorrt_demos](https://github.com/jkjung-avt/tensorrt_demos), los cuales proveen una implementacion de la _yolo layer_ , la cual es requerida para poder transformar yolov4 a TensorRT, y tambien, scripts de transformacion de modelos, especificamente de darknet a onnx, y onnx a TensorRT.

Con esto en mente, se modificó el detector de TensorRT de tal forma que fuera modular y facil de utilizar, este modulo se llama trt_yolo.py, el cual tiene una clase detector la cual basta con incluirla, instanciarla, y ejecutarla.


##### Seguimiento

Para el seguimiento, se pretenden utilizar dos tecnicas. Seguimiento [Naive](https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/), el cual consta de un seguidor en base a distancia euclidiana, y [DeepSort](https://arxiv.org/abs/1703.07402), el cual utiliza una capa de extraccion de caracteristicas para mantener una id. Ambos metodos utilizan como entrada una bbox de algun detector.

Para el metodo Naive, se implementa la version de PyimageSearch la cual se encuentra junto al detector de tensorrt en tracker_demo.py

##### Conteo

El conteo se llevara a cabo cuando un centroide de algun tracker cruza una linea. Este objeto linea mantendra toda la información con respecto a la gente que sube y que baja. Tambien mantendra conteo de las ids que ya cruzaron la linea y a su vez, en caso de que un centroide se "teletransporte" (es decir, desaparezca una id y aparezca en otro lado), no lo considerará en caso de problemas de asignacion de ID. 

Un demo del pipeline completo se encuentra en el archivo main.py


##### Otros trabajos

Tambien se adjunta una pequeña demo de un modulo que utiliza floodfill, se plantea como idea utilizar los stickers de la puerta para verificar si esta esta abierta o no, una demo de este algoritmo se encuentra en floodfill.py