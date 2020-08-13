#ifndef VORTEX_KERNEL_CU
#define VORTEX_KERNEL_CU

#include <cuda.h>
#include <cuda_runtime.h>


__global__ void VortexIndicator(
    double *device_arr1, int symbol_count, int indic_len, int dataLength,
    int time_len, int columns, double *device_outval, int win_size, double *error_val)
{
    int blockID = blockIdx.y*gridDim.x + blockIdx.x; /// find block Id in 2D grid of 1D blocks
	int idx = (threadIdx.x + (blockID*blockDim.x)); /// compute thread index for blockId
	if (idx < indic_len)
	{
		int symbol_idx = idx / dataLength; /// find symbol index from thread Idx
		int symbolOffset = symbol_idx * dataLength; /// compute each symbol start offset in 1D array

		int date_idx = (idx - symbolOffset) / time_len; 
		int dateOffset = date_idx*time_len; /// compute each date start offset
		int time_idx = idx - symbolOffset - dateOffset; /// compute time Idx for symbol and date from thread Idx
        int index = symbolOffset + dateOffset + time_idx; 

        int cIdx = index * columns;
        int outIdx = index * columns;

        // skip all error values and values less then average win size
        if(device_arr1[cIdx] == *error_val || device_arr1[cIdx+1] == *error_val ||
            device_arr1[cIdx+2] == *error_val ||(win_size == 0 && time_idx == 0))
        {
            device_outval[outIdx] = *error_val;
            device_outval[outIdx+1] = *error_val;
            device_outval[outIdx+2] = *error_val;
        }
        else
        {
            int count = 0;
            double plusVM = 0;
            double minusVM = 0;
            double trueRange = 0;
            double high = *error_val;
            double low = *error_val;

            for(int i=index; i >= symbolOffset; i--)
            {
                cIdx = i * columns;
                if(device_arr1[cIdx] != *error_val &&
                    device_arr1[cIdx+1] != *error_val &&
                    device_arr1[cIdx+2] != *error_val)
                {
                    if(high != *error_val && low != *error_val)
                    {
                        plusVM += abs(high - device_arr1[cIdx+1]); // current high - previous low
                        minusVM += abs(low - device_arr1[cIdx]); // current low - previous high

                        double val = max(high-low, abs(high-device_arr1[cIdx+2]));
                        trueRange += max(val, abs(low-device_arr1[cIdx+2]));
                        count++;
                    }

                    high = device_arr1[cIdx];
                    low = device_arr1[cIdx+1];

                    if(count == win_size)
                        break;
                }
            }

            if(count == win_size)
            {
                device_outval[outIdx] = trueRange;
                device_outval[outIdx+1] = minusVM / trueRange;
                device_outval[outIdx+2] = plusVM / trueRange;
            }
            else
            {
                device_outval[outIdx] = *error_val;
                device_outval[outIdx+1] = *error_val;
                device_outval[outIdx+2] = *error_val;
            }
        }
    }
}

#endif