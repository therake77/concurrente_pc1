import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class BitonicSorterThreads {
    private final int numThreads;

    public BitonicSorterThreads(int numThreads) {
        if (numThreads < 1) {
            throw new IllegalArgumentException("El numero de hilos debe ser mayor a 0");
        }
        this.numThreads = numThreads;
    }

    public void sort(double[] array) throws InterruptedException {
        int n = array.length;
        if (n == 0) return;
        if ((n & (n - 1)) != 0) {
            throw new IllegalArgumentException("El numero de elementos debe ser potencia de 2");
        }
        for (int k = 2; k <= n; k *= 2) {
            for (int j = k/2; j > 0; j /= 2) {
                parallelPhase(array, n, k, j);
            }
        }
    }

    private void parallelPhase(double[] array, int n, int k, int j) throws InterruptedException {
        Thread[] threads = new Thread[numThreads];
        int chunkSize = (n + numThreads - 1) / numThreads;

        for (int t = 0; t < numThreads; t++) {
            final int start = t * chunkSize;
            final int end = Math.min(start + chunkSize, n);
            threads[t] = new Thread(() -> {
                for (int i = start; i < end; i++){
                    int partner = i ^ j;
                    if (partner > i) {
                        boolean ascending = ((i & k) == 0);
                        compareAndSwap(array, i, partner, ascending);
                    }
                }
            });
            threads[t].start();
        }
        for (Thread thread : threads) {
            thread.join();
        }
    }

    private void compareAndSwap(double[] array, int i, int j, boolean ascending) {
        if (ascending) {
            if (array[i] > array[j]) {
                double temp = array[i];
                array[i] = array[j];
                array[j] = temp;
            }
        } else {
            if (array[i] < array[j]) {
                double temp = array[i];
                array[i] = array[j];
                array[j] = temp;
            }
        }
    }

    public static void printArray(double[] array) {
        for (double value : array) {
            System.out.print(value + " ");
        }
        System.out.println();
    }

    private double[] readArray(String ruta, int k) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(ruta));
        String linea;
        double[] array = new double[Math.powExact(2, k)];
        int i = 0;
        while ((linea = br.readLine()) != null && i < array.length) {
            array[i++] = Double.parseDouble(linea);
        }
        br.close();
        return array;
    }

    public static void main(String[] args) throws InterruptedException, IOException {
        int k = 25;
        int h = 6;
        BitonicSorterThreads sorter = new BitonicSorterThreads(h);
        double[] array = sorter.readArray("joaquin\\lista.txt", k);
        //System.out.println("Original array:");
        //printArray(array);
        sorter.sort(array);
        System.out.println("Sorted array");
        //printArray(array);
    }
}
