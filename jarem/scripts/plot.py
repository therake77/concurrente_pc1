from turtle import color

import matplotlib.pyplot as plt
def plot_mlp():
    n_cores = [1,2,4,6,8,10,12,14,16]
    training_times = [ 53126, 27692, 14291, 11333, 11338, 10288, 8912, 7931, 8015  ]
    inference_times = [ 344, 180, 90, 74, 62, 57, 50, 53, 60 ]
    gpu_training_time = 248 
    gpu_inference_time = 1 
    plt.plot(n_cores,training_times, color='blue') #type: ignore
    plt.plot(n_cores,inference_times, color='red') #type: ignore
    plt.axhline(gpu_training_time,color='green') #type: ignore
    plt.axhline(gpu_inference_time,color='orange') #type: ignore
    plt.show() #type: ignore

def plot_cnn()
    n_cores = [1,2,4,8,12,16]
    training_times = [ 43131 ]
    inference_times = [  ]
    gpu_training_time = 248 
    gpu_inference_time = 1

plot_mlp()