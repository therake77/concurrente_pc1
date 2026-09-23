# Hilos Java
Para ejecutar el algoritmo bitonic sort con hilos de Java, primero ejecutar Generator.java y luego BitonicSorterThreads.java.

# Cuda
Primero se debe tener instalado nvcc. Luego, dentro del directorio joaquin/ ejecutar:
```bash
nvcc --ptx bitonic_sort.cu -o bitonic_sort.ptx
```

Luego, crear un directorio joaquin/lib/ y añadir los archivos jcuda-12.6.0.jar y jcuda-natives-12.6.0-windows-x86_64.jar, o descargar el de sus sistema operativo correspondiente.

Para descargar ambos archivos, visite las paginas `https://repo1.maven.org/maven2/org/jcuda/jcuda/12.6.0/` y `https://repo1.maven.org/maven2/org/jcuda/jcuda-natives/12.6.0/`.

Luego, dependiendo de su IDE, agregue ambos archivos a las bibliotecas/librerias del proyecto.