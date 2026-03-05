#ifndef CL_HELPERS_H
#define CL_HELPERS_H

#include <CL/cl.h>
#include <stdio.h>      /* printf*/
#include <stdlib.h>		/* exit*/

const char *get_error_string_from_cl_error(cl_int error)
{
    switch(error){
            // run-time and JIT compiler errors
            case 0: return "CL_SUCCESS";
            case -1: return "CL_DEVICE_NOT_FOUND";
            case -2: return "CL_DEVICE_NOT_AVAILABLE";
            case -3: return "CL_COMPILER_NOT_AVAILABLE";
            case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
            case -5: return "CL_OUT_OF_RESOURCES";
            case -6: return "CL_OUT_OF_HOST_MEMORY";
            case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
            case -8: return "CL_MEM_COPY_OVERLAP";
            case -9: return "CL_IMAGE_FORMAT_MISMATCH";
            case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
            case -11: return "CL_BUILD_PROGRAM_FAILURE";
            case -12: return "CL_MAP_FAILURE";
            case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
            case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
            case -15: return "CL_COMPILE_PROGRAM_FAILURE";
            case -16: return "CL_LINKER_NOT_AVAILABLE";
            case -17: return "CL_LINK_PROGRAM_FAILURE";
            case -18: return "CL_DEVICE_PARTITION_FAILED";
            case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

            // compile-time errors
            case -30: return "CL_INVALID_VALUE";
            case -31: return "CL_INVALID_DEVICE_TYPE";
            case -32: return "CL_INVALID_PLATFORM";
            case -33: return "CL_INVALID_DEVICE";
            case -34: return "CL_INVALID_CONTEXT";
            case -35: return "CL_INVALID_QUEUE_PROPERTIES";
            case -36: return "CL_INVALID_COMMAND_QUEUE";
            case -37: return "CL_INVALID_HOST_PTR";
            case -38: return "CL_INVALID_MEM_OBJECT";
            case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
            case -40: return "CL_INVALID_IMAGE_SIZE";
            case -41: return "CL_INVALID_SAMPLER";
            case -42: return "CL_INVALID_BINARY";
            case -43: return "CL_INVALID_BUILD_OPTIONS";
            case -44: return "CL_INVALID_PROGRAM";
            case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
            case -46: return "CL_INVALID_KERNEL_NAME";
            case -47: return "CL_INVALID_KERNEL_DEFINITION";
            case -48: return "CL_INVALID_KERNEL";
            case -49: return "CL_INVALID_ARG_INDEX";
            case -50: return "CL_INVALID_ARG_VALUE";
            case -51: return "CL_INVALID_ARG_SIZE";
            case -52: return "CL_INVALID_KERNEL_ARGS";
            case -53: return "CL_INVALID_WORK_DIMENSION";
            case -54: return "CL_INVALID_WORK_GROUP_SIZE";
            case -55: return "CL_INVALID_WORK_ITEM_SIZE";
            case -56: return "CL_INVALID_GLOBAL_OFFSET";
            case -57: return "CL_INVALID_EVENT_WAIT_LIST";
            case -58: return "CL_INVALID_EVENT";
            case -59: return "CL_INVALID_OPERATION";
            case -60: return "CL_INVALID_GL_OBJECT";
            case -61: return "CL_INVALID_BUFFER_SIZE";
            case -62: return "CL_INVALID_MIP_LEVEL";
            case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
            case -64: return "CL_INVALID_PROPERTY";
            case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
            case -66: return "CL_INVALID_COMPILER_OPTIONS";
            case -67: return "CL_INVALID_LINKER_OPTIONS";
            case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

            // extension errors
            case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
            case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
            case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
            case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
            case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
            case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
            default: return "Unknown OpenCL error";
    }
}


void read_cl_kernel_from_file(char* file_path, char*& parsed_kernel_source, size_t& file_size_in_byte) {
    FILE* kernel_file =  fopen(file_path, "rb"); //open file in read and binary mode (to read file specifying num bytes), c-style

    if(!kernel_file) { //could not open kernel file, print error message and abort the program
        printf("\n");
        printf("Could not open kernel file: %s \n", file_path);
        printf("Please check your paths and directories!\n");
        printf("Aborting the program.\n");
        exit(-1);
    }

    fseek(kernel_file, 0, SEEK_END); // set the file cursor to the end of the file
    file_size_in_byte = ftell(kernel_file); //get the file cursor position in num bytes
    rewind(kernel_file); //reset file cursor to position 0 for reading from the beginning

    printf("reading %lu byte shader code\n", file_size_in_byte);

    parsed_kernel_source = (char*) malloc(file_size_in_byte + 1); // allocate enough memory to hold num chars in file + 1 (null termination)
    fread(parsed_kernel_source, 1, file_size_in_byte, kernel_file); //read the content of the kernel_file
    fclose(kernel_file);

    parsed_kernel_source[file_size_in_byte] = 0; // signal end of C-Style character array by setting the last byte to zero

    file_size_in_byte += 1;
}


cl_program compile_cl_kernel_with_error_log(cl_device_id const& device, cl_context const& context, char* kernel_source_path) {


    cl_program program = 0;
    char* stereo_matching_kernel_source = NULL;

    size_t num_bytes_in_cstring = 0;

    read_cl_kernel_from_file(kernel_source_path, stereo_matching_kernel_source, num_bytes_in_cstring);

    //printf("Kernel source: %s \n", stereo_matching_kernel_source);



    cl_int kernel_program_creation_status = 0;
    program = clCreateProgramWithSource(context, 1, (const char**)&stereo_matching_kernel_source, &num_bytes_in_cstring, &kernel_program_creation_status);
    if(kernel_program_creation_status != CL_SUCCESS)
    {
        printf("Error: clCreateProgram error\n");
        return 0;
    }

    kernel_program_creation_status = clBuildProgram(program, 1, &device ,NULL,NULL,NULL);
    if(kernel_program_creation_status != CL_SUCCESS)
    {
        printf("Error: clBuildProgram error\n");

        char *buff_erro;
        cl_int errcode;
        size_t build_log_len;
        errcode = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_len);
        if (errcode) {
                    printf("clGetProgramBuildInfo failed at line %d\n", __LINE__);
                    exit(-1);
                }

            buff_erro = (char*)malloc(build_log_len);
            if (!buff_erro) {
                printf("malloc failed at line %d\n", __LINE__);
                exit(-2);
            }

            errcode = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, build_log_len, buff_erro, NULL);
            if (errcode) {
                printf("clGetProgramBuildInfo failed at line %d\n", __LINE__);
                exit(-3);
            }

            fprintf(stderr,"Build log: \n%s\n", buff_erro); //Be careful with  the fprint
            free(buff_erro);
            fprintf(stderr,"clBuildProgram failed\n");
            exit(EXIT_FAILURE);
            return 0;
    }




    printf("Done compiling program from source: %s\n", kernel_source_path);

    return program;
    //stereo_matching_cl_program = ;
}



#endif //CL_HELPERS_H