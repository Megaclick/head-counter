# head-counter

## Introduccion 

Este repositorio contiene el trabajo hecho para el ramo de Vision por Computador [IPD-441] de la universidad Federico Santamaría.

El objetivo de este repositorio es entregar una herramienta para hacer seguimiento de personas y el conteo de salida y entrada de personas para evitar la evasión en buses.

Se prueban tanto Camshift, Sort, Deepsort y Distancia euclidiana utilizando Darknet y TensorRT.


![til](./images/testing.gif)

## Requerimientos
* CUDA >= 10.0: https://developer.nvidia.com/cuda-toolkit-archive 
* OpenCV https://opencv.org/releases/
* cuDNN >= 7.0 https://developer.nvidia.com/rdp/cudnn-archive 
* TensorRT > 7.2 https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html


## Compilación



```bash
git clone https://github.com/Megaclick/head-counter
cd head-counter
git submodule update --init
cp ./assets/darknet_api.py ./thirdparty/darknet/
```
Se recomienda instalar las dependencias de python dentro de un entorno virtual. Para ello seguir los siguientes pasos

```bash
sudo apt-get -y install virtualenv
virtualenv --python=python3.6 venv
source venv/bin/activate
pip3 install --no-use-pep517 -r requirements.txt
```
En caso de tener problemas con Tensorflow, se recomienda cambiar la version a una que sea compatible con la version de CUDA instalada.

En este trabajo se utilizaron dos tipos de detectores. es por ello que la compilación será distinta para cada uno de ellos. Estos detectores no son mutualmente excluyentes, por lo que el repositorio puede funcionar con solo uno de ellos.

#### Darknet
```bash
cd ./thirdparty/darknet
```
Se requiere modificar el makefile de la siguiente forma:

```makefile
GPU=1
CUDNN=1
CUDNN_HALF=1
OPENCV=1
AVX=0
OPENMP=0
LIBSO=1
ZED_CAMERA=0
ZED_CAMERA_v2_8=0
```
y a su vez, descomentar dependiendo del tipo de gpu que contenga el host. Por ejemplo, para la serie 2000 de nvidia descomentar la siguiente linea:

```makefile
ARCH= -gencode arch=compute_75,code=[sm_75,compute_75]
```
Luego
```bash
make -j$(nproc)
```

## TensorRT

Se requiere una versión compilada de TensorRT en el sistema, y a su vez, su version de python instalada dentro del entorno virtual. 
Luego, se debe cambiar dentro del Makefile de la carpeta plugins en la linea 30 y 31 el PATH donde estan las librerias de TensorRT.

Luego, dentro de la carpeta ./plugins ejecutar 
```bash
make
cp libyolo_layer.so ..
```

Después, se deberá hacer la conversion de pesos, para ello, seguir los siguientes pasos.
```bash
cd ./yolo
#copiar los cfg y weights a la carpeta yolo
cp ../models/head/yolov4-tiny-head-416.* .
python3 yolo_to_onnx.py -c 1 -m yolov4-tiny-head-416
python3 onnx_to_tensorrt.py -c 1 -m yolov4-tiny-head-416
```

## Run

Se dan a disposicion 8 archivos en donde se prueba Darknet/TensorRT junto con D.E, Camshift, Sort y Deepsort. Para la ejecucion se requieren 3 argumentos.

* -i --input Path del video de input
* -f --fskip Número de cuantos frames saltar
* -o --output Path del video de output

Para ejecutar los archivos tomar como referencia el siguiente comando
```bash
#python3 centroid/camshift/sort/deepsort_darknet/trt.py -i [input video path] -f [frames to skip] -o [output path]
python3 camshift_darknet.py -i ./videos/prueba2.mp4 -f 0 -o demo.avi
```

##### Otros trabajos

Tambien se adjunta una pequeña demo de un modulo que utiliza floodfill, se plantea como idea utilizar los stickers de la puerta para verificar si esta esta abierta o no, una demo de este algoritmo se encuentra en floodfill.py

```bash
python3 floodfill.py
```