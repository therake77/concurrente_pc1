# Práctica 1 - CC4P1

El código de la carpeta ```piero``` contiene git submodules, así que al clonar este repositorio usar:
```shell
git clone --recursive URL_DEL_REPO
cd piero/SampleRepo
git submodule update --remote --merge
```
Ejecutar con:
```
cd piero/SampleRepo
xmake r fem
```