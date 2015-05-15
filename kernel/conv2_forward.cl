__kernel void conv2_forward(const int map_in_num,
							const int map_in_side_len,
							const int map_out_num,
							const int map_out_side_len,
							const int kernel_side_len,
				__global float * W_acc, 
				__global float * B_acc,
				__global float * in_acc,
				__global float * out_acc
				)
{
		int l=get_global_id(0);
		int m=get_global_id(1);
		int n=get_global_id(2);
		float sum=0;
		int out_index=l*map_out_side_len*map_out_side_len+m*map_out_side_len+n;
		for(int i=0;i<map_in_num;i++)
		{
			for(int j=0;j<kernel_side_len;j++)
			{
				for(int k=0;k<kernel_side_len;k++)
				{
					sum+=W_acc[l*map_in_num*kernel_side_len*kernel_side_len+i*kernel_side_len*kernel_side_len+j*kernel_side_len+k]*in_acc[i*map_in_side_len*map_in_side_len+(m+j)*map_in_side_len+(n+k)];
				}
			}
		}
		sum+=B_acc[l];
		out_acc[out_index]=1/(1+exp(-sum));

}
