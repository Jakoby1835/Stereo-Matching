#include "cl_helpers.h"
#include "cl_kernel_launcher.h"

#include "project_types.h"

#include <algorithm>
#include <math.h>
#include <stdlib.h>     /* abs */

#include <limits.h>
#include <math.h>     /*sqrt*/

#include <omp.h>
#include <mpi.h>

#include <stdint.h>
#include <string>

#include <iostream>


#define WRITE_RECEIVED_DEBUG_IMAGE_CHUNKS_FOR_WORKERS 1

// print more debug information
#define VERBOSE_PROJECT 1

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"



void write_grayscale_bmp_image(uint8_t * image_buffer, char* file_name, int image_width, int image_height) {
	stbi_write_bmp(file_name, image_width, image_height, 1, image_buffer);
}


// this code needs to be ported to the stereo matching kernel in the cl-file - what is a straight-forward way to do it? can it be improved?
void compute_disp_map_ref_and_omp(unsigned char* left_image, unsigned char* right_image, char* result_image,
	image_dims const& dims, int window_half_size, int const max_disp) {

	int total_rows = dims.height - 2 * window_half_size;   // Anzahl der tatsächlich zu verarbeitenden Zeilen
	// Hinweis: Die Fortschrittsausgabe wurde in der parallelen Version entfernt,
	// da sie bei gleichzeitiger Ausführung mehrerer Threads zu inkonsistenten Ausgaben führen würde.
	// Stattdessen kann man bei Bedarf eine thread-sichere Fortschrittsanzeige implementieren,
	// z.B. mit einer atomaren Variable und periodischer Ausgabe durch den Master-Thread.

	// Parallelisierung der äußeren Schleife über die Zeilen.
	// Jeder Thread erhält einen Bereich von y-Werten und bearbeitet diese vollständig.
	// schedule(dynamic) wird verwendet, da die Arbeitslast pro Zeile variieren kann
	// (abhängig von der Position x und der Suchbreite für die Disparität).
	// Variablen, die von allen Threads gelesen werden (left_image, right_image, dims, ...)
	// sind implizit shared, da sie außerhalb der parallelen Region deklariert sind.
	// Alle innerhalb der Schleife deklarierten Variablen sind automatisch privat pro Thread.
//#pragma omp parallel for schedule(dynamic)
	for (int y = window_half_size; y < dims.height - window_half_size; ++y) {
		// Master‑Thread gibt alle 10 Zeilen seine aktuelle y‑Position aus
		if (omp_get_thread_num() == 0 && y % 10 == 0) {
			std::cout << "Master bei Zeile " << y
				<< " (ca. " << (100 * y / (dims.height - window_half_size)) << "%)\r";
			std::cout.flush();
		}
		for (int x = window_half_size; x < dims.width - window_half_size; ++x) {
			float best_NCC_cost = -1.0f;               // initialer Kostenwert (NCC liegt zwischen -1 und 1)
			int best_disparity_hypothesis = 0;          // beste bisher gefundene Disparität

			/* - Suche entlang der horizontalen Scanline (Epipolarlinie) im rechten Bild
			 *   nach dem Pixel, dessen Intensität am besten mit der Intensität des aktuellen Pixels
			 *   im linken Bild übereinstimmt.
			 * - Als Kostenfunktion wird die normalisierte Kreuzkorrelation (NCC) verwendet.
			 */

			 // Begrenzung des Suchbereichs basierend auf der maximalen Disparität
			int e_limit_left_hand_side = window_half_size;
			int e_limit_right_hand_side = window_half_size;
			if (x > max_disp) {
				e_limit_left_hand_side = std::max(window_half_size, (x - max_disp));                     // linke Grenze
				e_limit_right_hand_side = std::min((dims.width - 1) - window_half_size, x + max_disp); // rechte Grenze
			}

			// Iteration über alle möglichen Disparitäten (d.h. über mögliche Zielpixel im rechten Bild)
			for (int e = e_limit_left_hand_side; e <= e_limit_right_hand_side; ++e) {
				float current_NCC_cost = 0.0f;
				float ref_window_sum = 0.0f;
				float search_window_sum = 0.0f;
				int counter_value = 0;
				float variance_ab = 0.0f;
				float variance_a = 0.0f;
				float variance_b = 0.0f;

				// Summiere über das Fenster der Größe (2*window_half_size+1)^2
				for (int win_y = -window_half_size; win_y <= window_half_size; ++win_y) {
					for (int win_x = -window_half_size; win_x <= window_half_size; ++win_x) {
						int search_index = ((y + win_y) * dims.width + e + win_x);
						int reference_index = ((y + win_y) * dims.width + x + win_x);

						ref_window_sum += left_image[reference_index];
						search_window_sum += right_image[search_index];

						variance_ab += left_image[reference_index] * right_image[search_index];
						variance_a += left_image[reference_index] * left_image[reference_index];
						variance_b += right_image[search_index] * right_image[search_index];

						++counter_value;
					}
				}

				// Berechnung der Mittelwerte und Varianzen für die NCC-Formel
				float ref_window_mean = ref_window_sum / counter_value;
				float search_window_mean = search_window_sum / counter_value;
				variance_ab /= counter_value;
				variance_a /= counter_value;
				variance_b /= counter_value;

				// Normalized Cross-Correlation (NCC)
				current_NCC_cost = (variance_ab - ref_window_mean * search_window_mean)
					/ std::sqrt((variance_a - ref_window_mean * ref_window_mean) *
						(variance_b - search_window_mean * search_window_mean));

				// Höhere NCC-Werte (näher an 1) bedeuten bessere Übereinstimmung
				if (current_NCC_cost > best_NCC_cost) {
					best_NCC_cost = current_NCC_cost;
					/* - Berechne die aktuelle beste Disparität
					 *   basierend auf der Position des Referenzpixels (links) und
					 *   der Position des momentan besten Suchpixels (rechts).
					 */
					best_disparity_hypothesis = (x - e);
				}
			}

			// Schreibe das Ergebnis in das Ausgabebild
			int pixel_offset = (y * dims.width + x);
			result_image[pixel_offset] = static_cast<char>(best_disparity_hypothesis);
		}
	} // Ende der parallelisierten Schleife
}

// function which compares the consistency of the results of the left and the right disparity maps
void left_right_consistency_check(char* left_disp_image, char* right_disp_image, char* left_consistency_checked, image_dims const& dims, float valid_disparity_limit) {

#pragma omp parallel for collapse(2)
	for (int y = 0; y < dims.height; ++y) {
		for (int x = 0; x < dims.width; ++x) {

			int image_read_idx_left = x + y * dims.width;

			int left_disparity = left_disp_image[image_read_idx_left];

			int neighbor_horizontal_position = x + left_disparity;
			if (neighbor_horizontal_position < 0 || neighbor_horizontal_position >= dims.width) {
				left_disp_image[image_read_idx_left] = 127; // out of bounds, invalidate with value 255
			}
			else {
				int image_read_idx_right = neighbor_horizontal_position + y * dims.width;
				int right_disparity = right_disp_image[image_read_idx_right];

				int recovered_left_position = neighbor_horizontal_position + right_disparity;

				if (abs(recovered_left_position - x) < valid_disparity_limit) {
					left_disp_image[image_read_idx_left] = left_disp_image[image_read_idx_left];
				}
				else {
					left_disp_image[image_read_idx_left] = 127; // recovered position is too far off, set to 255 (invalidate)
				}
			}

		}
	}


}


void setup_cl_environment(cl_device_id& device, cl_context& context, int worker_rank, int highest_worker_rank) {

	printf("Setting up OpenCL-Environment... (worker %d / %d).\n", worker_rank, highest_worker_rank);
	cl_platform_id platform;
	cl_command_queue queue;
	cl_int err;

	cl_int status = clGetPlatformIDs(1, &platform, NULL);

	if (CL_SUCCESS != status) {
		printf("Could not find a suitable CL platform, aborting the program (worker %d / %d).\n", worker_rank, highest_worker_rank);
		exit(-1);
	}

	cl_uint num_available_GPUs = 0;

	clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_available_GPUs);



	if (num_available_GPUs == 0)    //no GPU available. fallback: pick a CPU (not efficient, but works)
	{
		printf("No GPU device available (worker %d / %d).\n", worker_rank, highest_worker_rank);
		printf("Choosing CPU as default device (worker %d / %d).\n", worker_rank, highest_worker_rank);

		cl_uint num_available_CPUs = 0;

		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &num_available_CPUs);
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
	}
	else
	{
		/* just pick the first GPU that we can find.
		   For multi-GPU systems, one could check for several GPUs and pick one,
		   or have all of them working at the same time */
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

		printf("Found GPU (worker %d / %d).\n", worker_rank, highest_worker_rank);
	}


	// create the context (i.e. the CL states) associated with the picked device
	context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);

	printf("Done setting up OpenCL-Environment (worker %d / %d).\n", worker_rank, highest_worker_rank);
}



int main(int argc, char* argv[])
{

	/* example command lines:
		mpirun -n 1 ./StereoMatcher ./images/rect1.png ./images/rect2.png 0 0        //"no openmp/single threaded openmp", no GPGPU via openCL, 1 MPI worker (default implementation, should work for you without writing anything)
		mpirun -n 1 ./StereoMatcher ./images/rect1.png ./images/rect2.png 1 0        // all available openMP threads for computation, no GPGPU via openCL    (should have openmp added in fitting parts)
		./StereoMatcher ./images/rect1.png ./images/rect2.png 1 1        // all available openMP threads for computation, + stereo matching on the GPU       (should have opencl kernels implemented properly)
		
		NOTE: If you are working under WINDOWS and therore using MS-MPI, prelace 'mpirun'  with 'mpiexec' in the command line example above!!!
	
	*/
	/*if (argc < 5) {
		printf("USAGE: %s left_img_filename[path] right_img_filename[path] USE_OPENMP[0|1] USE_OPENCL[0|1]\n", argv[0]);
		return 0;
	}*/


	// default settings which are valid even if MPI is not used
	// -> not using openMPI is basically the same as using one MPI node
	int rank = 0;
	int num_mpi_nodes = 1;

	// initialize MPI in any case (even if we only work with one node -> simplifies code flow )
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_mpi_nodes);



	printf("Rank: %d, #MPI-Nodes: %d\n", rank, num_mpi_nodes);
	printf("After MPI initialization\n");


	int const highest_mpi_node_rank = num_mpi_nodes - 1;

	// read input parameters
	char* const left_img_name = "..\\images\\rect1.png";//argv[1];
	char* const right_img_name = "..\\images\\rect2.png";//argv[2];
	bool use_openmp = 0;//std::atoi(argv[3]);
	bool use_opencl = 1;//std::atoi(argv[4]);


	if (use_openmp) {
		int num_threads_to_use = omp_get_max_threads();
		omp_set_num_threads(num_threads_to_use);
	}
	else {
		omp_set_num_threads(1);
	}




	int complete_image_width = 0;
	int complete_image_height = 0;
	int num_channels = 0;
	int forced_num_channels = 1; //implicit grayscale conversion

	// define image pointers a bit more global such that we can address these pointers later on as "working_cpu_buffer_..."
	unsigned char* left_image = NULL;
	unsigned char* right_image = NULL;

	// let master node read the entire image // the workers will get the image chunks they need via MPI_Scatterv
	if (0 == rank) {
		//master loads complete input images 
		left_image = stbi_load(left_img_name, &complete_image_width, &complete_image_height, &num_channels, 1);    //load RGB image as 1 channel grayscale image (1x unsigned 8 bit per pixel)
		right_image = stbi_load(right_img_name, &complete_image_width, &complete_image_height, &num_channels, 1);  //load RGB image as 1 channel grayscale image (1x unsigned 8 bit per pixel)
	}
	printf("Image Loaded\n"); // trying to find the 
	if (num_mpi_nodes > 1) {
		// the master should broadcast the original image sizes, such that every worker can compute their block sizes on their own broadcast image sizes
		MPI_Bcast(&complete_image_width, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&complete_image_height, 1, MPI_INT, 0, MPI_COMM_WORLD);
	}

	// store the original image dimensions
	image_dims full_image_dimensions{ complete_image_width, complete_image_height };
	// gets replaced if MPI is used and will contain the height of the image chunks
	image_dims working_dimensions{ complete_image_width, complete_image_height };


	float const FULL_SIZE_IMAGE_WIDTH = 2872;


	float const stereo_matching_parameter_downscaling = (full_image_dimensions.width / FULL_SIZE_IMAGE_WIDTH);
	// defines the region around the image windows, which is (2*window_halfsize + 1) ^ 2, i.e. a window halfsize creates square window with a side length of (2*5 + 1) pixels = 11 pixels
	int const window_halfsize = 10 * stereo_matching_parameter_downscaling;
	// for this kind of aerial image, the matching search window is either to the left or to the right of the reference image. we check max_disparity positions in both the positive and negative direction
	int const max_disparity = 80 * stereo_matching_parameter_downscaling;




	// define source location for stereo matching
	// this will be overridden 
	unsigned char* working_cpu_buffer_image_source_left = left_image;
	unsigned char* working_cpu_buffer_image_source_right = right_image;





	// the loaded images serve as source for filling the CL buffers, except if we use MPI, then we let the pointers point to somewhere else
	// (note that the pointer is either way only valid for the node with id 0)


	// image parts that have to be matched can be mentally divided into core rows, for which disparity values need to be computet, top padding rows, and bottom padding rows
	// in the openmpi part, we calculate the number of bytes to send for each of the rows for each worker (including the master), and also calculate the read offsets in the
	// original image parts
	int const SIZEOF_INT = sizeof(int);
	int* core_rows_send_counts = (int*)malloc(SIZEOF_INT * num_mpi_nodes);
	int* core_row_send_offset_original_image_buffer = (int*)malloc(SIZEOF_INT * num_mpi_nodes);

	int* top_padding_rows_send_counts = (int*)malloc(SIZEOF_INT * num_mpi_nodes);
	int* top_padding_row_send_offset_original_image_buffer = (int*)malloc(SIZEOF_INT * num_mpi_nodes);

	int* bottom_padding_rows_send_counts = (int*)malloc(SIZEOF_INT * num_mpi_nodes);
	int* bottom_padding_row_send_offset_original_image_buffer = (int*)malloc(SIZEOF_INT * num_mpi_nodes);

	// we will free these lists later after we gathered the disparity map chunks from the workers, because we may want to reuse some of our calculations here

	if (num_mpi_nodes > 1) {
		printf("complete image width: %d as known by worker with id: %d\n", complete_image_width, rank);
		printf("complete image height: %d as known by worker with id: %d\n", complete_image_height, rank);

		int const num_non_overlapping_rows_per_worker = (complete_image_height + num_mpi_nodes - 1) / num_mpi_nodes;
		int const num_non_overlapping_rows_last_worker = complete_image_height - num_non_overlapping_rows_per_worker * (num_mpi_nodes - 1);


		// compute send count and read offset for each mpi node.
		// careful! stereo matching needs vertical overlap between image stripes (exception at first and last node)

		//for simplicity we assume that our window size is not larger than the image
		unsigned int num_non_overlapping_rows_assigned = 0;
		for (int node_idx = 0; node_idx < num_mpi_nodes; ++node_idx) {

			// precompute the values for the calls for scatterv and gatherv for all nodes 
			// master node
			if (0 == node_idx) {
				core_rows_send_counts[node_idx] = (num_non_overlapping_rows_per_worker /*+ window_halfsize*/)* complete_image_width;
				core_row_send_offset_original_image_buffer[node_idx] = 0;

				top_padding_rows_send_counts[node_idx] = 0; // top part should have no top padding, so send size is 0 byte ...
				top_padding_row_send_offset_original_image_buffer[node_idx] = 0;  // ... and we just pick 0 as read offset, although we do not really read anything

				bottom_padding_rows_send_counts[node_idx] = window_halfsize * complete_image_width; // top part should have no top padding, so send size is 0 byte ...
				bottom_padding_row_send_offset_original_image_buffer[node_idx] = core_row_send_offset_original_image_buffer[node_idx] + core_rows_send_counts[node_idx];  // add the core row offset block to the core read offset to arrive at the starting position for the lower padding

			}
			else if (node_idx == num_mpi_nodes - 1) { //last node with unique row count + one sided overlap

				core_rows_send_counts[node_idx] = (num_non_overlapping_rows_last_worker /*+ window_halfsize*/)* complete_image_width;
				core_row_send_offset_original_image_buffer[node_idx] = (num_non_overlapping_rows_per_worker)* complete_image_width * (node_idx); //- window_halfsize * complete_image_width;

				top_padding_rows_send_counts[node_idx] = window_halfsize * complete_image_width; // top part should have no top padding, so send size is 0
				top_padding_row_send_offset_original_image_buffer[node_idx] = core_row_send_offset_original_image_buffer[node_idx] - window_halfsize * complete_image_width;  //top part sould have no top padding, so we just choose the read offset of 0

				bottom_padding_rows_send_counts[node_idx] = 0; // top part should have no top padding, so send size is 0 byte ...
				bottom_padding_row_send_offset_original_image_buffer[node_idx] = 0;  // add the core row offset block to the core read offset to arrive at the starting position for the lower padding

			}
			else { //every worker that is not the first or the last one has two sided overlap

				core_rows_send_counts[node_idx] = (num_non_overlapping_rows_last_worker /*+ 2 * window_halfsize*/)* complete_image_width;
				core_row_send_offset_original_image_buffer[node_idx] = (num_non_overlapping_rows_per_worker)* complete_image_width * (node_idx); //- window_halfsize * complete_image_width;

				top_padding_rows_send_counts[node_idx] = window_halfsize * complete_image_width; // top part should have no top padding, so send size is 0
				top_padding_row_send_offset_original_image_buffer[node_idx] = core_row_send_offset_original_image_buffer[node_idx] - window_halfsize * complete_image_width;  //top part sould have no top padding, so we just choose the read offset of 0

				bottom_padding_rows_send_counts[node_idx] = window_halfsize * complete_image_width; // top part should have no top padding, so send size is 0 byte ...
				bottom_padding_row_send_offset_original_image_buffer[node_idx] = core_row_send_offset_original_image_buffer[node_idx] + core_rows_send_counts[node_idx];  // add the core row offset block to the core read offset to arrive at the starting position for the lower padding

			}
		}



		// will contain non-overlapping pixels per worker
		unsigned int num_core_pixels_for_current_worker = 0;
		// will contain num pixels for top-part of the image (overlapping region)
		unsigned int num_top_pixels_for_current_worker = 0;
		// will contain num pixels for bottom-part of the image (overlapping region)
		unsigned int num_bottom_pixels_for_current_worker = 0;

		// same variables, but only contains ROWS!! instead of pixels!! (for writing debug images)
		unsigned int top_padding_number_of_rows_for_current_worker = 0;
		unsigned int core_number_of_rows_for_current_worker = 0;
		unsigned int bottom_padding_number_of_rows_for_current_worker = 0;

		// only account for overlap at the bottom of the  overlap for  sided overlap for master node
		if (0 == rank) {
			top_padding_number_of_rows_for_current_worker = 0;
			core_number_of_rows_for_current_worker = num_non_overlapping_rows_per_worker;
			bottom_padding_number_of_rows_for_current_worker = window_halfsize;
		}
		else if (rank == num_mpi_nodes - 1) { //last node with unique row count + one sided overlap
			top_padding_number_of_rows_for_current_worker = window_halfsize;
			core_number_of_rows_for_current_worker = num_non_overlapping_rows_last_worker;
			bottom_padding_number_of_rows_for_current_worker = 0;
		}
		else { //every worker that is not the first or the last one has two sided overlap
			top_padding_number_of_rows_for_current_worker = window_halfsize;
			core_number_of_rows_for_current_worker = num_non_overlapping_rows_per_worker;
			bottom_padding_number_of_rows_for_current_worker = window_halfsize;
		}


		int total_num_pixels_for_current_worker = (core_number_of_rows_for_current_worker + top_padding_number_of_rows_for_current_worker + bottom_padding_number_of_rows_for_current_worker) * complete_image_width;  //

		// we omit multiplying the number of bytes by sizeof(char) or sizeof(unsigned char), because this is 1
		num_core_pixels_for_current_worker = core_number_of_rows_for_current_worker * complete_image_width; // * sizeof(unsigned char);
		num_top_pixels_for_current_worker = top_padding_number_of_rows_for_current_worker * complete_image_width; // * sizeof(unsigned char);
		num_bottom_pixels_for_current_worker = bottom_padding_number_of_rows_for_current_worker * complete_image_width; // * sizeof(unsigned char);


		unsigned char* left_image_stride_for_current_worker = (unsigned char*)malloc(total_num_pixels_for_current_worker);
		unsigned char* right_image_stride_for_current_worker = (unsigned char*)malloc(total_num_pixels_for_current_worker);

		if (0 == rank) {
			for (int send_count_idx = 0; send_count_idx < num_mpi_nodes; ++send_count_idx) {
				printf("send count: %d for worker %d \n", core_rows_send_counts[send_count_idx], send_count_idx);
				printf("displacement count: %d for worker %d \n", core_row_send_offset_original_image_buffer[send_count_idx], send_count_idx);
			}
		}


		//careful! overlap of regions in one scatterv call is not allowed! therefore we call 3x scatterv: 1. distribute top overlap, 2. distribute core part, 3. distribute bottom overlap
		//https://cvw.cac.cornell.edu/mpicc/using-collective-communication/scatter-scatterv
		//https://cvw.cac.cornell.edu/mpicc/using-collective-communication/scatterv-syntax 
		
		/////////////////////////////////
		////// SCATTER TOP PADDING ROWS. 
		/////////////////////////////////
		// Bonus-TODO: SCATTER TOP PADDING ROWS WITH SCATTERV


		/////////////////////////
		////// SCATTER CORE ROWS. 
		/////////////////////////
		// Scatterv is needed because the chunks can be of different size for different workers, and in general we want to be able to choose arbitrary read offsets

		// send left and right image core parts. the offset image first call MPI_Scatterv to send the precomputed number of elements and right offsets for anyone

		//scatter left grayscale input image
		MPI_Scatterv(left_image, core_rows_send_counts, core_row_send_offset_original_image_buffer,
			MPI_BYTE,  // the input image is unsigned bytes [0, 255] . the disparity image needs to be sent back as signed bytes, since the disparity is signed , aka MPI_CHAR
			&left_image_stride_for_current_worker[num_top_pixels_for_current_worker], // skip the top padding pixels in the target image and start writing from there
			num_core_pixels_for_current_worker,  // read as many pixels as there are in the core block
			MPI_BYTE,  // see above
			0, MPI_COMM_WORLD); //distributes from the master (rank == 0) in the common world 

// scatter right input image
		MPI_Scatterv(right_image, core_rows_send_counts, core_row_send_offset_original_image_buffer,
			MPI_BYTE,  // the input image is unsigned bytes [0, 255] . the disparity image needs to be sent back as signed bytes, since the disparity is signed , aka MPI_CHAR
			&right_image_stride_for_current_worker[num_top_pixels_for_current_worker], // skip the top padding pixels in the target image and start writing from there
			num_core_pixels_for_current_worker, // read as many pixels as there are in the core block
			MPI_BYTE,  // see above
			0, MPI_COMM_WORLD);  //distributes from the master (rank == 0) in the common world 



///////////////////////////////////
////// SCATTER BOTTOM PADDING ROWS. 
///////////////////////////////////

//Bonus-TODO: : SCATTER BOTTOM PADDING ROWS WITH SCATTERV


		unsigned int total_number_of_rows_to_write_for_worker = top_padding_number_of_rows_for_current_worker + bottom_padding_number_of_rows_for_current_worker + core_number_of_rows_for_current_worker;

		// since we are using MPI, we use the received chunk as cpu source buffer for each worker ... 
		working_cpu_buffer_image_source_left = left_image_stride_for_current_worker;
		working_cpu_buffer_image_source_right = right_image_stride_for_current_worker;

		// ... and set the precomputed chunk sizes for use in all matching calculations ...
		working_dimensions.width = complete_image_width;
		working_dimensions.height = total_number_of_rows_to_write_for_worker;


#if WRITE_RECEIVED_DEBUG_IMAGE_CHUNKS_FOR_WORKERS
		//write the received debug images at every worker
		write_grayscale_bmp_image(left_image_stride_for_current_worker, (char*)(std::to_string(rank) + "_Left_" + ".bmp").c_str(), complete_image_width, total_number_of_rows_to_write_for_worker);
		write_grayscale_bmp_image(right_image_stride_for_current_worker, (char*)(std::to_string(rank) + "_Right_" + ".bmp").c_str(), complete_image_width, total_number_of_rows_to_write_for_worker);
#endif

	}



	// calculate the number of by pixels to match per worker based on the image chunk dimensions for every worker
	int64_t const num_byte_per_image_part_to_match = working_dimensions.width * working_dimensions.height * sizeof(char);

	// allocate memory chunk of this size
	char* disparity_image_chunk_left_to_right = (char*)malloc(num_byte_per_image_part_to_match);
	char* disparity_image_chunk_right_to_left = (char*)malloc(num_byte_per_image_part_to_match);


	//reference implementation & openMP branch
	if (!use_opencl) {
		printf("!use_opencl\n");
		compute_disp_map_ref_and_omp(working_cpu_buffer_image_source_left, working_cpu_buffer_image_source_right,
			disparity_image_chunk_left_to_right,
			working_dimensions, window_halfsize, max_disparity);


		compute_disp_map_ref_and_omp(working_cpu_buffer_image_source_right, working_cpu_buffer_image_source_left,
			disparity_image_chunk_right_to_left,
			working_dimensions, window_halfsize, max_disparity);

	}
	else { // if opencl should be used, this branch is executed for matchign rather than the reference / openmp one

	 //initialize OpenCL-related variables
	 /////////////////////////////////////
		cl_device_id device = NULL; // CL compute device
		cl_context context = NULL;  // CL context storing states associated with device
		setup_cl_environment(device, context, rank, highest_mpi_node_rank);

		cl_command_queue command_queue = command_queue = clCreateCommandQueue(context, device, 0, NULL);  // queue storing submitted commands and executing them in order

		// read and create the program (see cl_helpers.h)
		cl_program stereo_matching_cl_program = compile_cl_kernel_with_error_log(device, context, (char*)"../cl_kernels/stereo_matching_kernels.cl"); // GPGPU program
		// create the NCC stereo matching kernel
		cl_kernel stereo_matching_cl_kernel = clCreateKernel(stereo_matching_cl_program, "ncc_stereo_matching", NULL); // GPGPU kernel

		// create CL buffers for left image, right image, left disparity image and right disparity image
		////////////////////////////////////////////////////////////////////////////////////////////////
		// GPU representation of left image chunk/stride for worker -- should be interpreted as unsigned char* buffer in the kernel
		cl_mem cl_grayscale_image_left_hand_side = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_byte_per_image_part_to_match, (void*)working_cpu_buffer_image_source_left, NULL);  //directly copy from left grayscale CPU image
		// GPU representation of right image chunk/stride for worker -- should be interpreted as unsigned char* buffer in the kernel
		cl_mem cl_grayscale_image_right_hand_side = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_byte_per_image_part_to_match, (void*)working_cpu_buffer_image_source_right, NULL); //directly copy from right grayscale CPU image


		// GPU representation of the left disparity image chunk / stride that the worker computes -- should be interpreted as (signed) char* buffer in the kernel, since the matching pixels can be found to the left or to the right
		cl_mem cl_disparity_image_left_to_right = clCreateBuffer(context, CL_MEM_WRITE_ONLY, num_byte_per_image_part_to_match, NULL, NULL); // dont fill with source data, because we create the content in the CL kernel
		// GPU representation of the right disparity image chunk / stride that the worker computes -- should be interpreted as (signed) char* buffer in the kernel, since the matching pixels can be found to the left or to the right
		cl_mem cl_disparity_image_right_to_left = clCreateBuffer(context, CL_MEM_WRITE_ONLY, num_byte_per_image_part_to_match, NULL, NULL);



		// perform stereo matching left to right and read result back into cpu buffer disparity_image_left_to_image  //  compute_disp_map_cl
		{
			// see cl_kernel_launcher.h
			launch_stereo_matching_kernel(command_queue, stereo_matching_cl_kernel, // command queue and NCC stereo matching kernel
				cl_grayscale_image_left_hand_side, cl_grayscale_image_right_hand_side, // left and right image chunks to be stereo matched
				cl_disparity_image_left_to_right,  // cl buffer containing result of disparity matching from left to right image
				working_dimensions,  // 2d dimensions of an image chunk -> global work group sizes
				window_halfsize, max_disparity); // parameters for the stereo matching


/* Read the disparity put back to host memory.*/
			cl_int status = clEnqueueReadBuffer(command_queue, cl_disparity_image_left_to_right, CL_TRUE, 0, num_byte_per_image_part_to_match, disparity_image_chunk_left_to_right, 0, NULL, NULL);

			if (CL_SUCCESS != status) {
				printf("clEnqueueReadBuffer for transferring left disparity map from gpu to cpu memory failed (@ worker id %d / %d)\n", rank, num_mpi_nodes);
			}

		}


		// perform stereo matching right to left and read result back into cpu buffer disparity_image_right_to_image   //  compute_disp_map_cl
		{
			// see cl_kernel_launcher.h
			launch_stereo_matching_kernel(command_queue, stereo_matching_cl_kernel, // command queue and NCC stereo matching kernel
				cl_grayscale_image_right_hand_side, cl_grayscale_image_left_hand_side, // right and left image chunks to be stereo matched
				cl_disparity_image_right_to_left,  // cl buffer containing result of disparity matching from right to left image
				working_dimensions,  // 2d dimensions of an image chunk -> global work group sizes
				window_halfsize, max_disparity); // parameters for the stereo matching

// Read the disparity right to left and copy it back from device to host memory.
			cl_int status = clEnqueueReadBuffer(command_queue, cl_disparity_image_right_to_left, CL_TRUE, 0, num_byte_per_image_part_to_match, disparity_image_chunk_right_to_left, 0, NULL, NULL);

			if (CL_SUCCESS != status) {
				printf("clEnqueueReadBuffer for transferring right disparity map from gpu to cpu memory failed (@ worker id %d / %d)\n", rank, num_mpi_nodes);
			}

		}

	}



	// TODO: GATHER NON-OVERLAPPING DISPARITY IMAGE PARTS IF MPI IS USED

	//////////////////////////////////////////////////////////////
	///////OPENMPI: GATHER THE DISPARITY RESULTS FROM EVERY PARTY
	//////////////////////////////////////////////////////////////

	// if we do not use openmp, the chunk is the entire image and we just reference this, otherwise we allocate a new disparity image with full size and collect the results from the nodes
	char* complete_disparity_image_left_to_right = disparity_image_chunk_left_to_right;
	char* complete_disparity_image_right_to_left = disparity_image_chunk_right_to_left;



	if (num_mpi_nodes > 1) {
		//we have to allocate new memory (disparity maps of full image size) and collect the results
		complete_disparity_image_left_to_right = (char*)malloc(full_image_dimensions.width * full_image_dimensions.height);
		complete_disparity_image_right_to_left = (char*)malloc(full_image_dimensions.width * full_image_dimensions.height);


		int* receive_counts = (int*)malloc(sizeof(int) * num_mpi_nodes);
		int* offset_counts = (int*)malloc(sizeof(int) * num_mpi_nodes);

		//TODO: COMPUTE PARAMETERS FOR MPI_Gatherv

		//TODO: CALL MPI_Gatherv TO RECEIVE signed 8-bit disparity image parts left-to-right
		//TODO: CALL MPI_Gatherv TO RECEIVE signed 8-bit disparity image parts right-to-left

		// free additional int arrays we created for scatterv
		free(receive_counts);
		free(offset_counts);

		// also finally free leftover arrays from configuration of scatterv (because we reused some of them)
		free(core_rows_send_counts);
		free(core_row_send_offset_original_image_buffer);
		free(top_padding_rows_send_counts);
		free(top_padding_row_send_offset_original_image_buffer);
		free(bottom_padding_rows_send_counts);
		free(bottom_padding_row_send_offset_original_image_buffer);
	}




	// use the results from stereo matching, perform left right consistency check, map the colors, write the results to a bmp file

	// the remaining left right consistency check is lightweight compared to the matching task. we can just let the master do this for the entire image, although it would also be fine to perform on the workers
	if (0 == rank) {

		float valid_disparity_limit = 20.0 * stereo_matching_parameter_downscaling;
		left_right_consistency_check(complete_disparity_image_left_to_right, complete_disparity_image_right_to_left, NULL, full_image_dimensions, valid_disparity_limit);


		unsigned int const num_output_channels = 3;
		unsigned char* disparity_image_vis_8bit_left_right = (unsigned char*)malloc(complete_image_width * complete_image_height * num_output_channels * sizeof(unsigned char));

		// the master should write the final disparity image
		for (int pixel_idx = 0; pixel_idx < full_image_dimensions.width * full_image_dimensions.height; ++pixel_idx) {

			unsigned int const disparity_image_read_idx = pixel_idx;

			float disparity_normalization_factor = 127.0 * (stereo_matching_parameter_downscaling);

			unsigned int const visualized_image_write_offset = pixel_idx * num_output_channels;
			if (complete_disparity_image_left_to_right[disparity_image_read_idx] != 127) {
				for (int pixel_channel_idx = 0; pixel_channel_idx < num_output_channels; ++pixel_channel_idx) {
					disparity_image_vis_8bit_left_right[visualized_image_write_offset + pixel_channel_idx] = (complete_disparity_image_left_to_right[disparity_image_read_idx] / (-disparity_normalization_factor) + 0.5) * 127.0;
				}
			}
			else {
				disparity_image_vis_8bit_left_right[visualized_image_write_offset + 0] = 0;
				disparity_image_vis_8bit_left_right[visualized_image_write_offset + 1] = 0;
				disparity_image_vis_8bit_left_right[visualized_image_write_offset + 2] = 255;
			}
		}



		std::string const output_filename = std::string("final_disparity_map__")
			+ "width_" + std::to_string(full_image_dimensions.width) + "__"
			+ "height_" + std::to_string(full_image_dimensions.height) + "__"
			+ "mp_" + (use_openmp ? "enabled" : "disabled") + "__"
			+ "mpinodes_" + std::to_string(num_mpi_nodes) + "__"
			+ "cl_" + (use_opencl ? "enabled" : "disabled")
			+ ".bmp";


		//write out the final result 
		//Note: Output values should be of type unsigned char 
		stbi_write_bmp(output_filename.c_str(), complete_image_width, complete_image_height, num_output_channels, disparity_image_vis_8bit_left_right);

		printf("\n\nWrote resulting disparity map to \"./%s\" \n\n", output_filename.c_str());


		//dealloc dynamic memory -- master allocated the complete images, so only the master deallocates them 
		free(disparity_image_vis_8bit_left_right);
		stbi_image_free(left_image);
		stbi_image_free(right_image);
	}


	if (num_mpi_nodes > 1) {
		free(complete_disparity_image_left_to_right);
		free(complete_disparity_image_right_to_left);
	}


	MPI_Finalize();


	return 0;
}