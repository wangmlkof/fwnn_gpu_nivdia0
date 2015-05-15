#include <OpenCL/opencl.h>
#include <stdio.h>
#include <time.h>
#include <iostream>
#include <string>
#include <stdlib.h>
#include "myclutils.h"

//A(m*n) * B(n*p) = C(m*p)
#define MDIM 1000
#define NDIM 1000
#define PDIM 1000
using namespace std;

static float A[MDIM*NDIM];
static float B[NDIM*PDIM];
static float C[MDIM*PDIM];

int main(int argc, char * argv[])
{
		cl_context context;
		cl_device_id device;
		cl_int errNum;
		cl_command_queue command;
		cl_program program;
		cl_kernel kernel=0;
		cl_mem memory[3]={0,0,0};
		int Mdim=MDIM;
		int Ndim=NDIM;
		int Pdim=PDIM;
		//create context for device of gpu type
		context=CreateGPUContext();
		//create command queue for the first device on gpu context
		command=CreateCommandQueue(context,&device,0);
		//create program object for that device
		//program=CreateProgram(context,device,"mat_mul1.cl");
		//program=CreateProgram(context,device,"mat_mul2.cl");
		program=CreateProgram(context,device,"mat_mul3.cl");
		//create kernel (an instance for that program)
		kernel=clCreateKernel(program,"mat_mul",NULL);
		
		//initial A  matrix
		for(int i=0;i<Mdim;i++)
		{
				for(int j=0;j<Ndim;j++)
				{
						A[i*Ndim+j]=1;
				}
		}	
		//initial B  matrix
		for(int i=0;i<Ndim;i++)
		{
				for(int j=0;j<Pdim;j++)
				{
						B[i*Pdim+j]=2;
				}
		}	
		//create memory Objects for storing arguments in kernel function
		memory[0]=clCreateBuffer(context,CL_MEM_READ_ONLY ,sizeof(float)*Mdim*Ndim,NULL,NULL);
		memory[1]=clCreateBuffer(context,CL_MEM_READ_ONLY ,sizeof(float)*Ndim*Pdim,NULL,NULL);
		memory[2]=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*Mdim*Pdim,NULL,NULL);
		errNum = clEnqueueWriteBuffer(command, memory[0], CL_TRUE, 0,sizeof(float) * Mdim*Ndim, A, 0, NULL, NULL);
		errNum = clEnqueueWriteBuffer(command, memory[1], CL_TRUE, 0,sizeof(float) * Ndim*Pdim, B, 0, NULL, NULL);
		//set arguments for kernel functions
		errNum=0;
		errNum |= clSetKernelArg(kernel, 0, sizeof(int),&Mdim);
		errNum |= clSetKernelArg(kernel, 1, sizeof(int),&Ndim);
		errNum |= clSetKernelArg(kernel, 2, sizeof(int),&Pdim);
		errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem),&memory[0]);
		errNum |= clSetKernelArg(kernel, 4, sizeof(cl_mem),&memory[1]);
		errNum |= clSetKernelArg(kernel, 5, sizeof(cl_mem),&memory[2]);

		int start=clock();
		//size_t global[2]={(size_t)Mdim,(size_t)Pdim};
		//errNum = clEnqueueNDRangeKernel(command, kernel, 2, NULL,global,NULL,0, NULL, NULL);

		size_t global[1]={(size_t)Mdim};
		size_t local[1]={(size_t)25};
		errNum = clEnqueueNDRangeKernel(command, kernel, 1, NULL, global , local, 0, NULL, NULL);

		clFinish(command);
		int end=clock();
		printf("running time is %f s\n",1.0*(end-start)/CLOCKS_PER_SEC);
//		cout<<"running time is "<<1.0*(end-start)/CLOCKS_PER_SEC<<" s"<<endl;

		errNum = clEnqueueReadBuffer(command, memory[2],CL_TRUE, 0, sizeof(float)*Mdim*Pdim,C, 0, NULL, NULL);
		for(int i=0;i<Mdim*Pdim;i++)
		{
				if(C[i]!=2000)
				{
						cout<<"error result:"<<i<<" "<<C[i]<<endl;
						return 0;
				}
		}
		return 0;
}
