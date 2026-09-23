extern "C"
__global__ void bitonicPhase(
    double* array,
    int n,
    int k,
    int j
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    int partner = i ^ j;

    if (partner > i) {
        bool ascending = ((i & k) == 0);

        double current = array[i];
        double other = array[partner];

        if (ascending) {
            if (current > other) {
                array[i] = other;
                array[partner] = current;
            }
        } else {
            if (current < other) {
                array[i] = other;
                array[partner] = current;
            }
        }
    }
}