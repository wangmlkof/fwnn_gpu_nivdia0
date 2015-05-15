__kernel void conv1_backward_delta(
							const int map_in_num,
							const int map_in_side_len,
							const int map_out_num,
							const int map_out_side_len,
							const int kernel_side_len,
				__global float * W_acc, 
				__global float * in_acc,
				__global float * delta_in_acc,
				__global float * delta_out_acc)
{
		int i=get_global_id(0);
		int m=get_global_id(1);
		int n=get_global_id(2);

		for(int j=0;j<kernel_side_len;j++)
		{
			for(int k=0;k<kernel_side_len;k++)
			{
				int in_index=i*map_in_side_len*map_in_side_len+(m+j)*map_in_side_len+(n+k);
				delta_out_acc[in_index]=0;
				for(int l=0;l<map_out_num;l++)
				{
					float w_temp=W_acc[l*map_in_num*kernel_side_len*kernel_side_len+i*kernel_side_len*kernel_side_len+j*kernel_side_len+k];
					int out_index=l*map_out_side_len*map_out_side_len+m*map_out_side_len+n;
					delta_out_acc[in_index]+=delta_in_acc[out_index]*w_temp;
				}
			}
		}
}
