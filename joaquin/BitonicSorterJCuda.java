import jcuda.Pointer;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUevent;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;

import static jcuda.driver.JCudaDriver.*;

public class BitonicSorterJCuda {
    private CUcontext context;
    private CUmodule module;
    private CUfunction bitonicPhase;

    public BitonicSorterJCuda() {
        setExceptionsEnabled(true);

        cuInit(0);

        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);

        context = new CUcontext();
        cuCtxCreate(context, 0, device);

        module = new CUmodule();
        cuModuleLoad(module, "joaquin/bitonic_sort.ptx");

        bitonicPhase = new CUfunction();

        cuModuleGetFunction(bitonicPhase, module, "bitonicPhase");
    }

    public SortTiming sortWithTiming(double[] array, int threadsPerBlock) {
        int n = array.length;

        if (n == 0) {
            return new SortTiming(0.0, 0.0);
        }
        if ((n & (n - 1)) != 0) {
            throw new IllegalArgumentException("El numero de elementos debe ser potencia de 2");
        }
        if (threadsPerBlock <= 0 || threadsPerBlock > 1024) {
            throw new IllegalArgumentException("threadsPerBlock debe estar entre 1 y 1024");
        }

        CUdeviceptr d_array = new CUdeviceptr();
        long bytes = (long) n * Double.BYTES;

        // Tiempo total desde Java
        long totalStart = System.nanoTime();
        cuMemAlloc(d_array, bytes);

        try {
            // Java -> GPU
            cuMemcpyHtoD(d_array, Pointer.to(array), bytes);
            int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
            /*
             * Eventos CUDA para medir solamente
             * la ejecución de los kernels.
             */
            CUevent kernelStart = new CUevent();
            CUevent kernelStop = new CUevent();

            cuEventCreate(kernelStart, 0);
            cuEventCreate(kernelStop, 0);

            cuEventRecord(kernelStart, null);

            // Bitonic Sort
            for (int k = 2; k <= n; k *= 2) {
                for (int j = k / 2; j > 0; j /= 2) {
                    Pointer kernelParameters =
                            Pointer.to(
                                    Pointer.to(d_array),
                                    Pointer.to(new int[]{n}),
                                    Pointer.to(new int[]{k}),
                                    Pointer.to(new int[]{j})
                            );

                    cuLaunchKernel(
                            bitonicPhase,
                            blocks, 1, 1,
                            threadsPerBlock, 1, 1,
                            0,
                            null,
                            kernelParameters,
                            null
                    );
                    /*
                     * Cada fase debe terminar antes
                     * de comenzar la siguiente.
                     */
                    cuCtxSynchronize();
                }
            }

            cuEventRecord(kernelStop, null);
            cuEventSynchronize(kernelStop);

            float kernelMilliseconds = getElapsedTime(kernelStart, kernelStop);

            cuEventDestroy(kernelStart);
            cuEventDestroy(kernelStop);

            // GPU -> Java
            cuMemcpyDtoH(Pointer.to(array), d_array, bytes);

            long totalEnd = System.nanoTime();
            double totalMilliseconds = (totalEnd - totalStart) / 1_000_000.0;

            return new SortTiming(totalMilliseconds, kernelMilliseconds);
        } finally {
            cuMemFree(d_array);
        }
    }

    public void sort(double[] array, int threadsPerBlock) {
        sortWithTiming(array, threadsPerBlock);
    }

    public void sort(double[] array) {
        sort(array, 256);
    }

    public void close() {
        if (context != null) {
            cuCtxDestroy(context);
            context = null;
        }
    }

    public static boolean isSorted(double[] array) {
        for (int i = 1; i < array.length; i++) {
            if (array[i - 1] > array[i]) {
                return false;
            }
        }
        
        return true;
    }

    public static class SortTiming {
        private final double totalMilliseconds;
        private final double kernelMilliseconds;

        public SortTiming(double totalMilliseconds, double kernelMilliseconds) {
            this.totalMilliseconds = totalMilliseconds;
            this.kernelMilliseconds = kernelMilliseconds;
        }

        public double getTotalMilliseconds() {
            return totalMilliseconds;
        }

        public double getKernelMilliseconds() {
            return kernelMilliseconds;
        }
    }

    private static float getElapsedTime(CUevent start, CUevent stop) {
        float[] milliseconds = new float[1];
        cuEventElapsedTime(milliseconds, start, stop);
        return milliseconds[0];
    }
}