__kernel
void stencil(__global float *input, __global float *output, __global float *filter, int cols) {
    int i = get_global_id(0); // <--- this is the row id
    int j = get_global_id(1);

    float temp = 0.0;
    for (int k1 = 0; k1 < 3; k1++) {
        for (int k2 = 0; k2 < 3; k2++) {
            // doing this
            //mul = filter[k1*3 + k2] * orig[(r+k1)*col_size + c+k2];
            temp += filter[k1 * 3 + k2] * input[(i + k1) * cols + j + k2];
        }
    }
    output[i * cols + j] = temp;
}
