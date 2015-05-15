__kernel void hidden1_backward_dWB(const int input_size,
				__global float * dW_acc, 
				__global float * dB_acc,
				__global float * in_acc,
				__global float * out_acc,
				__global float * delta_in_acc)
{
		int k=get_global_id(0);
		delta_in_acc[k]*=out_acc[k]*(1-out_acc[k]);
		for(int j=0;j<input_size;j++)
		{
			dW_acc[k*input_size+j]+=in_acc[j]*(-delta_in_acc[k]);
		}
		dB_acc[k]+=(-delta_in_acc[k]);
}
