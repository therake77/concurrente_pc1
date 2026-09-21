import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Random;

public class Generator {
    private final int k;

    public Generator(int k) {
        this.k = k;
    }

    private void generate() {
        try {
            try (FileWriter archivo = new FileWriter("joaquin\\lista.txt")) {
                archivo.flush();
            }
            System.out.println("Archivo creado");
        } catch (IOException e) {
            System.out.println("Error al crear el archivo");
            e.printStackTrace();
        }
    }

    private void write(double[] array) {
        try {
            PrintWriter writer = new PrintWriter("joaquin\\lista.txt");
            for (int i = 0; i < array.length; i++) {
                writer.println(array[i]);

            }
            writer.close();
            System.out.println("Archivo escrito");
        } catch (IOException e) {
            System.out.println("Error al escribir el archivo");
            e.printStackTrace();
        }
    }

    public void start() {
        generate();
        Random rnd = new Random();
        double[] array = new double[Math.powExact(2, k)];
        for (int i = 0; i < array.length; i++) {
            array[i] = rnd.nextDouble(10000000);
        }
        write(array);
    }

    public static void main(String[] args) {
        Generator gen = new Generator(25);
        gen.start();
    }
}