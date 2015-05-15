__kernel void hidden1_backward_delta(const int input_size,
								 	 const int output_size,
				__global float * W_acc, 
				__global float * in_acc,
				__global float * delta_in_acc,
				__global float * delta_out_acc)
{
		int k=get_global_id(0);
		delta_out_acc[k]=0;
		for(int m=0;m<output_size;m++)
		{
			delta_out_acc[k]+=delta_in_acc[m]*W_acc[m*input_size+k];
		}	
}
