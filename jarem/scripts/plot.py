
import matplotlib.pyplot as plt


def plot(
    x : list[int],
    y : list[int],
    title : str,
    x_label : str,
    y_label : str,
):
    plt.plot(x,y,'g-')
    plt.scatter(x,y,color='red')
    plt.suptitle(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()



class Dataset:
    n_cores : list[int]
    training_time : list[int]
    inference_time : list[int]
    gpu_training_time : int
    gpu_inference_time : int

    def __init__(
            self,
            p0 : list[int],
            p1 : list[int],
            p2 : list[int],
            p3 : int,
            p4 : int
        ) -> None:

        self.n_cores = p0  
        self.training_time = p1 
        self.inference_time = p2
        self.gpu_training_time = p3 
        self.gpu_inference_time = p4

    def plot_training(self, title : str):
        plot(
            self.n_cores,
            self.training_time,
            title,
            "# CPU cores",
            "Miliseconds (ms)"
        )
    def plot_inference(self, title : str):
            plot(
                self.n_cores,
                self.inference_time,
                title,
                "# CPU cores",
                "Miliseconds (ms)"
            )

    
        
if __name__ == "__main__":
    n_cores = [1,2,4,8,12,16,20]

    mlp_training_times = [ 53103, 27302, 14145, 11747, 9172, 7860, 9756  ]
    mlp_inference_times = [ 338, 171, 88, 73, 56, 53, 55 ]
    mlp_gpu_training_time = 258
    mlp_gpu_inference_time = 1

    cnn_training_times = [ 167178, 88159, 48691, 36829, 31510, 28920, 34195 ]
    cnn_inference_times = [ 1269, 712, 425, 301, 285, 261, 278 ]
    cnn_gpu_training_time = 2095
    cnn_gpu_inference_time = 10

    mlp_dataset = Dataset(
        n_cores,
        mlp_training_times,
        mlp_inference_times,
        mlp_gpu_training_time,
        mlp_gpu_inference_time
    )

    cnn_dataset = Dataset(
        n_cores,
        cnn_training_times,
        cnn_inference_times,
        cnn_gpu_training_time,
        cnn_gpu_inference_time
    )

    #mlp_dataset.plot_training("MLP - Training (AMD Ryzen 7 5700X)")
    #mlp_dataset.plot_inference("MLP - Inference (AMD Ryzen 7 5700X)")
    #cnn_dataset.plot_training("CNN - Training (AMD Ryzen 7 5700X)")
    cnn_dataset.plot_inference( "CNN - Inference (AMD Ryzen 7 5700X)" )
