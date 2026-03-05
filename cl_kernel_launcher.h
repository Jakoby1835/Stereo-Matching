#ifndef CL_KERNEL_LAUNCHER_H
#define CL_KERNEL_LAUNCHER_H

#include "project_types.h"

#include <CL/cl.h>
#include <stdio.h>      /* printf*/


void launch_stereo_matching_kernel(cl_command_queue const& queue, cl_kernel const& stereo_matching_kernel, cl_mem const& cl_image_chunk_left, cl_mem const& cl_image_chunk_right, cl_mem const& cl_disparity_chunk_l_to_r,
								   image_dims const& matching_dimensions, int window_half_size, int max_disparity) {
        // perform stereo matching left to right and read result back into cpu buffer disparity_image_left_to_image
        {

            // upload all the CL kernel parameters for the left to right stereo matching
            if( CL_SUCCESS != clSetKernelArg(stereo_matching_kernel, 0, sizeof(cl_mem), &cl_image_chunk_left)  )        { printf("clSetKernelArg could not set kernel argument 0\n");}
            if( CL_SUCCESS != clSetKernelArg(stereo_matching_kernel, 1, sizeof(cl_mem), &cl_image_chunk_right) )        { printf("clSetKernelArg could not set kernel argument 1\n");}
            if( CL_SUCCESS != clSetKernelArg(stereo_matching_kernel, 2, sizeof(cl_mem), &cl_disparity_chunk_l_to_r) )   { printf("clSetKernelArg could not set kernel argument 2\n");}
            if( CL_SUCCESS != clSetKernelArg(stereo_matching_kernel, 3, sizeof(cl_int), &matching_dimensions.width) )   { printf("clSetKernelArg could not set kernel argument 3\n");}
            if( CL_SUCCESS != clSetKernelArg(stereo_matching_kernel, 4, sizeof(cl_int), &matching_dimensions.height) )  { printf("clSetKernelArg could not set kernel argument 4\n");}
            if( CL_SUCCESS != clSetKernelArg(stereo_matching_kernel, 5, sizeof(cl_int), &window_half_size) )            { printf("clSetKernelArg could not set kernel argument 5\n");}
            if( CL_SUCCESS != clSetKernelArg(stereo_matching_kernel, 6, sizeof(cl_int), &max_disparity) )               { printf("clSetKernelArg could not set kernel argument 6\n");}
     
            size_t global_work_size[2] = {size_t(matching_dimensions.width), size_t(matching_dimensions.height) };

            // enqueues the kernel based on the code in ./cl_kernels/stereo_matching_kernels.cl
            cl_int status = clEnqueueNDRangeKernel(queue, stereo_matching_kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);

            if(CL_SUCCESS != status) {
                printf("clEnqueueNDRangeKernel failed: \n");
                printf("CL error: %s\n", get_error_string_from_cl_error(status) ); 
            }

        }
}

#endif // CL_KERNEL_LAUNCHER_H