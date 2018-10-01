//OpenCL utility layer include
#include "xcl2.hpp"
#include <iostream>
#include <vector>
#include <stdlib.h>

#define COLS 5000
#define DATA_SIZE COLS*COLS
#define FILTER_SIZE 9
#define EPSILON (0.000001)

using namespace std;

bool stencil_comp(vector<float,aligned_allocator<float>> in, vector<float,aligned_allocator<float>> out, vector<float,aligned_allocator<float>> filter, vector<float,aligned_allocator<float>> fpga_out);

int main(int argc, char** argv){
	srand(148667);

    //Allocate Memory in Host Memory
    size_t vector_size_bytes = sizeof(float) * DATA_SIZE;
    size_t filter_size_bytes = sizeof(float) * FILTER_SIZE;

    //Initialize inputs
    std::vector<float,aligned_allocator<float>> input  (DATA_SIZE);
    std::vector<float,aligned_allocator<float>> filter  (FILTER_SIZE);
    std::vector<float,aligned_allocator<float>> output (DATA_SIZE);
    std::vector<float,aligned_allocator<float>> ref_output (DATA_SIZE);

    for (int i = 0; i < DATA_SIZE; i++){
         input[i] = ((float)rand()/((float)RAND_MAX))*1000000.0;
         output[i] = 0;
         ref_output[i]=0;
    }

    for (int i = 0; i < FILTER_SIZE; i++){
    	filter[i] = 1;
    }

    //OPENCL HOST CODE AREA START
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];
    cl::Context context(device);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);
    std::string device_name = device.getInfo<CL_DEVICE_NAME>();

    //Create Program and Kernel
    std::string binaryFile = xcl::find_binary_file(device_name,"stencil");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    cl::Program program(context, devices, bins);

    cl::Kernel krnl_stencil(program,"stencil");

    //Allocate Buffer in Global Memory
    cl::Buffer buffer_input (context, CL_MEM_READ_ONLY,
                        vector_size_bytes);

    cl::Buffer buffer_filter (context, CL_MEM_READ_ONLY,
    					filter_size_bytes);

    cl::Buffer buffer_output(context, CL_MEM_WRITE_ONLY, 
                        vector_size_bytes);

    //Copy input data to device global memory
    //change this to CL_FALSE potentially
    q.enqueueWriteBuffer(buffer_input, CL_TRUE, 0, vector_size_bytes, input.data());
    q.enqueueWriteBuffer(buffer_filter, CL_TRUE, 0, filter_size_bytes, filter.data());

    // int inc = INCR_VALUE;
    int size = DATA_SIZE;
    int cols = COLS;

    //Set the Kernel Arguments
    int narg=0;
    krnl_stencil.setArg(narg++,buffer_input);
    krnl_stencil.setArg(narg++,buffer_output);
    krnl_stencil.setArg(narg++,buffer_filter);
    krnl_stencil.setArg(narg++,cols);

    //Launch the Kernel
    q.enqueueNDRangeKernel(krnl_stencil,cl::NullRange,cl::NDRange(cols-2,(size/cols)-2),cl::NullRange);

    //Copy Result from Device Global Memory to Host Local Memory
    q.enqueueReadBuffer(buffer_output, CL_TRUE, 0, vector_size_bytes, output.data());
    q.finish();

    if(stencil_comp(input, ref_output, filter, output)) {
    	std::cout << "PASSED" << std::endl;
    }else{
    	std::cout << "FAILED" << std::endl;
    }
    return 0;
}

bool stencil_comp(vector<float,aligned_allocator<float>> input, vector<float,aligned_allocator<float>> out, vector<float,aligned_allocator<float>> filter, vector<float,aligned_allocator<float>> fpga_out) {
	/* Mach Suite Computation */
	int r, c, k1, k2;
	float temp, mul;
	for (r=0; r<COLS-2; r++) {
		for (c=0; c<COLS-2; c++) {
			temp = (float)0;
			for (k1=0;k1<3;k1++){
				for (k2=0;k2<3;k2++){
					mul = filter[k1*3 + k2] * input[(r+k1)*COLS + c+k2];
					temp += mul;
				}
			}
			out[(r*COLS) + c] = temp;
		}
	}

	for(int i=0; i<DATA_SIZE; i++) {
		if((fpga_out[i]-out[i])>EPSILON || (out[i]-fpga_out[i])>EPSILON) {return false;}
	}
	return true;
}


