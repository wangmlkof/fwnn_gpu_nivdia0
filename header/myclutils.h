#include <CL/opencl.h>
#include <string>
using namespace std;
cl_context CreateGPUContext();
cl_context CreateCPUContext();
cl_command_queue CreateCommandQueue(cl_context context,cl_device_id *device,int device_num);
cl_program CreateProgram(cl_context context, cl_device_id device,const char* fileName);
const char *getErrorString(cl_int error);
void DisplayPlatformInfo(cl_platform_id id, cl_platform_info name,string str);
void DisplayDeviceInfo(cl_device_id id,cl_device_info name,string str);
void GetDevices(cl_platform_id id);
