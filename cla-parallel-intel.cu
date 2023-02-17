/*********************************************************************/
//
// 02/02/2023: Revised Version for 32M bit adder with 32 bit blocks
//
/*********************************************************************/


#include "main-intel.h"
//#include "clockcycle.h"

//Touch these defines
#define input_size 8388608 // hex digits 
#define block_size 32
#define verbose 0

#define blocks 2048
#define threads_per_block 2048


//Do not touch these defines
#define digits (input_size+1)
#define bits (digits*4)
#define ngroups bits/block_size
#define nsections ngroups/block_size
#define nsupersections nsections/block_size
#define nsupersupersections nsupersections/block_size

//Global definitions of the various arrays used in steps for easy access
/***********************************************************************************************************/
// ADAPT AS CUDA managedMalloc memory - e.g., change to pointers and allocate in main function. 
/***********************************************************************************************************/
int gi[bits] = {0};
int pi[bits] = {0};
int ci[bits] = {0};

int ggj[ngroups] = {0};
int gpj[ngroups] = {0};
int gcj[ngroups] = {0};

int sgk[nsections] = {0};
int spk[nsections] = {0};
int sck[nsections] = {0};

int ssgl[nsupersections] = {0} ;
int sspl[nsupersections] = {0} ;
int sscl[nsupersections] = {0} ;

int sssgm[nsupersupersections] = {0} ;
int ssspm[nsupersupersections] = {0} ;
int ssscm[nsupersupersections] = {0} ;

int sumi[bits] = {0};

int sumrca[bits] = {0};

//Integer array of inputs in binary form
int* bin1=NULL;
int* bin2=NULL;

//Character array of inputs in hex form
char* hex1=NULL;
char* hex2=NULL;



/***************************************************************/
//making pointers to the host arrays - later, these will be cudaMalloc-ed
int * p_gi 	= gi;
int * p_pi 	= pi;
int * p_ci 	= ci;

int * p_ggj 	= ggj;
int * p_gpj 	= gpj;
int * p_gcj	= gcj;

int * p_sgk	= sgk;
int * p_spk	= spk;
int * p_sck	= sck;

int * p_ssgl	= ssgl;
int * p_sspl	= sspl;
int * p_sscl 	= sscl;

int * p_sssgm	= sssgm;
int * p_ssspm 	= ssspm;
int * p_ssscm	= ssscm;

int * p_sumi	= sumi;
/***************************************************************/

void read_input()
{
  char* in1 = (char *)calloc(input_size+1, sizeof(char));
  char* in2 = (char *)calloc(input_size+1, sizeof(char));

  if( 1 != scanf("%s", in1))
    {
      printf("Failed to read input 1\n");
      exit(-1);
    }
  if( 1 != scanf("%s", in2))
    {
      printf("Failed to read input 2\n");
      exit(-1);
    }
  
  hex1 = grab_slice_char(in1,0,input_size+1);
  hex2 = grab_slice_char(in2,0,input_size+1);
  
  free(in1);
  free(in2);
}
__device__
int* device_grab_slice(int* input, int starti, int length)//I needed this for the globalized functions that used grab_slice
{
  int* output;
  cudaMalloc(&output, length*sizeof(int));

    int i,j;
    for(i = 0, j = starti; i<length; i++,j++)
    {
        output[i] = input[j];
    }
    return output;
}



/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/


__global__
void global_compute_gp(int * p_gi,int *  p_pi,int *  bin1,int *  bin2)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < bits; i+=stride)
    {
        p_gi[i] = bin1[i] & bin2[i];
        p_pi[i] = bin1[i] | bin2[i];
    }
}

/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/


__global__
void global_compute_group_gp(int * p_gi, int * p_pi, int * p_ggj,int * p_gpj)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int j = index; j < ngroups; j+=stride)
    {
        int jstart = j * block_size;
        int* ggj_group = device_grab_slice(p_gi, jstart, block_size);
        int* gpj_group = device_grab_slice(p_pi, jstart, block_size);

        int sum = 0;
        for (int i = 0; i < block_size; i++)
        {
            int mult = ggj_group[i]; //grabs the g_i term for the multiplication
            for (int ii = block_size - 1; ii > i; ii--)
            {
                mult &= gpj_group[ii]; //grabs the p_i terms and multiplies it with the previously multiplied stuff (or the g_i term if first round)
            }
            sum |= mult; //sum up each of these things with an or
        }
        p_ggj[j] = sum;

        int mult = gpj_group[0];
        for (int i = 1; i < block_size; i++)
        {
            mult &= gpj_group[i];
        }
        p_gpj[j] = mult;

        // free from grab_slice allocation
        free(ggj_group);
        free(gpj_group);
    }
}

/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/

__global__
void global_compute_section_gp(int * p_ggj,int * p_gpj,int * p_sgk,int * p_spk)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int k = index; k < nsections; k+=stride)
    {
        int kstart = k * block_size;
        int* sgk_group = device_grab_slice(p_ggj, kstart, block_size);
        int* spk_group = device_grab_slice(p_gpj, kstart, block_size);

        int sum = 0;
        for (int i = 0; i < block_size; i++)
        {
            int mult = sgk_group[i];
            for (int ii = block_size - 1; ii > i; ii--)
            {
                mult &= spk_group[ii];
            }
            sum |= mult;
        }
        p_sgk[k] = sum;

        int mult = spk_group[0];
        for (int i = 1; i < block_size; i++)
        {
            mult &= spk_group[i];
        }
        p_spk[k] = mult;

        // free from grab_slice allocation
        free(sgk_group);
        free(spk_group);
    }
}

/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/


__global__
void global_compute_super_section_gp(int * p_sgk, int* p_spk, int* p_ssgl, int* p_sspl)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int l = index; l < nsupersections; l+=stride)
    {
        int lstart = l * block_size;
        int* ssgl_group = device_grab_slice(p_sgk, lstart, block_size);
        int* sspl_group = device_grab_slice(p_spk, lstart, block_size);

        int sum = 0;
        for (int i = 0; i < block_size; i++)
        {
            int mult = ssgl_group[i];
            for (int ii = block_size - 1; ii > i; ii--)
            {
                mult &= sspl_group[ii];
            }
            sum |= mult;
        }
        p_ssgl[l] = sum;

        int mult = sspl_group[0];
        for (int i = 1; i < block_size; i++)
        {
            mult &= sspl_group[i];
        }
        p_sspl[l] = mult;

        // free from grab_slice allocation
        free(ssgl_group);
        free(sspl_group);
    }
}
/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/


__global__
void global_compute_super_super_section_gp(int * p_ssgl,int * p_sspl,int * p_sssgm,int * p_ssspm)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(int m = index; m < nsupersupersections ; m+=stride)
    {
      int mstart = m*block_size;
      int* sssgm_group = device_grab_slice(p_ssgl,mstart,block_size);
      int* ssspm_group = device_grab_slice(p_sspl,mstart,block_size);
      
      int sum = 0;
      for(int i = 0; i < block_size; i++)
        {
	  int mult = sssgm_group[i];
	  for(int ii = block_size-1; ii > i; ii--)
            {
	      mult &= ssspm_group[ii];
            }
	  sum |= mult;
        }
      p_sssgm[m] = sum;
      
      int mult = ssspm_group[0];
      for(int i = 1; i < block_size; i++)
        {
	  mult &= ssspm_group[i];
        }
      p_ssspm[m] = mult;
      
      // free from grab_slice allocation
      free(sssgm_group);
      free(ssspm_group);
    }
}
/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/

__global__
void global_compute_super_super_section_carry(int * p_ssscm,int * p_sssgm,int * p_ssspm)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(int m = index; m < nsupersupersections; m+=stride)
    {
      int ssscmlast=0;
      if(m==0)
        {
	  ssscmlast = 0;
        }
      else
        {
	  ssscmlast = p_ssscm[m-1];
        }
      
      p_ssscm[m] = p_sssgm[m] | (p_ssspm[m]&ssscmlast);
    }
}

/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/


__global__
void global_compute_super_section_carry(int * p_ssscm,int *  p_sscl,int *  p_ssgl,int *  p_sspl)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
   int stride = blockDim.x * gridDim.x;
  for(int l = index; l < nsupersections; l+=stride)
    {
      int sscllast=0;
      if(l%block_size == block_size-1)
        {
	  sscllast = p_ssscm[l/block_size];
        }
      else if( l != 0 )
        {
	  sscllast = p_sscl[l-1];
        }
      
      p_sscl[l] = p_ssgl[l] | (p_sspl[l]&sscllast);
    }
}

/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/


__global__
void global_compute_section_carry(int * p_sscl,int * p_sck,int * p_sgk, int * p_spk)
{
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   int stride = blockDim.x * gridDim.x;
  for(int k = index; k < nsections; k+=stride)
    {
      int scklast=0;
      if(k%block_size==block_size-1)
        {
	  scklast = p_sscl[k/block_size];
        }
      else if( k != 0 )
        {
	  scklast = p_sck[k-1];
        }
      
      p_sck[k] = p_sgk[k] | (p_spk[k]&scklast);
    }
}


/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/


__global__
void global_compute_group_carry(int * p_sck, int * p_gcj, int * p_ggj, int * p_gpj)
{
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   int stride = blockDim.x * gridDim.x;
  for(int j = index; j < ngroups; j+= stride)
    {
      int gcjlast=0;
      if(j%block_size==block_size-1)
        {
	  gcjlast = p_sck[j/block_size];
        }
      else if( j != 0 )
        {
	  gcjlast = p_gcj[j-1];
        }
      
      p_gcj[j] = p_ggj[j] | (p_gpj[j]&gcjlast);
    }
}
/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/


__global__
void global_compute_carry(int * p_gi, int * p_pi, int * p_ci, int * p_gcj)
{
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   int stride = blockDim.x * gridDim.x;
  for(int i = index; i < bits; i+=stride)
    {
      int clast=0;
      if(i%block_size==block_size-1)
        {
	  clast = p_gcj[i/block_size];
        }
      else if( i != 0 )
        {
	  clast = p_ci[i-1];
        }
      
      p_ci[i] = p_gi[i] | (p_pi[i]&clast);
    }
}

/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/

__global__
void global_compute_sum(int * bin1, int * bin2, int * p_ci, int * p_sumi)
{
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   int stride = blockDim.x * gridDim.x;
  for(int i = index; i < bits; i+=stride)
    {
      int clast=0;
      if(i==0)
        {
	  clast = 0;
        }
      else
        {
	  clast = p_ci[i-1];
        }
      p_sumi[i] = bin1[i] ^ bin2[i] ^ clast;
    }
}

void cla(int * p_gi,int * p_pi,int * p_ci,  int *  p_ggj,int * p_gpj,int * p_gcj,  int *  p_sgk,int * p_spk,int * p_sck,  int *  p_ssgl,int * p_sspl,int * p_sscl,  int *  p_sssgm,int * p_ssspm,int * p_ssscm,   int * p_sumi,int * bin1,int * bin2)
{
  /***********************************************************************************************************/
  // ADAPT ALL THESE FUNCTUIONS TO BE SEPARATE CUDA KERNEL CALL
  // NOTE: Kernel calls are serialized by default per the CUDA kernel call scheduler
  // NOTE: Make sure you set the right CUDA Block Size (e.g., threads per block) for different runs per 
  //       assignment description.
  /***********************************************************************************************************/
    //compute_gp();
    global_compute_gp<<<blocks,threads_per_block>>>(p_gi, p_pi, bin1, bin2);
    cudaDeviceSynchronize();
    
    //compute_group_gp();
    global_compute_group_gp<<<blocks,threads_per_block>>>( p_gi, p_pi, p_ggj, p_gpj);
    cudaDeviceSynchronize();

    //compute_section_gp();
    global_compute_section_gp<<<blocks,threads_per_block>>>(p_ggj, p_gpj, p_sgk, p_spk);
    cudaDeviceSynchronize();

    //compute_super_section_gp();
    global_compute_super_section_gp<<<blocks,threads_per_block>>>(p_sgk, p_spk, p_ssgl, p_sspl);
    cudaDeviceSynchronize();

    //compute_super_super_section_gp();
    global_compute_super_super_section_gp<<<blocks, threads_per_block>>>( p_ssgl, p_sspl, p_sssgm, p_ssspm);
    cudaDeviceSynchronize();
    
    
    //compute_super_super_section_carry();
    global_compute_super_super_section_carry<<<blocks, threads_per_block>>>( p_ssscm, p_sssgm, p_ssspm);
    cudaDeviceSynchronize();
    
    //compute_super_section_carry();
    global_compute_super_section_carry<<<blocks, threads_per_block>>>(p_ssscm,p_sscl,p_ssgl,p_sspl);
    cudaDeviceSynchronize();
    
    //compute_section_carry();
    global_compute_section_carry<<<blocks, threads_per_block>>>(p_sscl,p_sck,p_sgk,p_spk);
    cudaDeviceSynchronize();
    
    //compute_group_carry();
    global_compute_group_carry<<<blocks, threads_per_block>>>(p_sck, p_gcj, p_ggj, p_gpj);
    cudaDeviceSynchronize();
    
    
    //compute_carry();
    global_compute_carry<<<blocks, threads_per_block>>>(p_gi, p_pi, p_ci, p_gcj);
    cudaDeviceSynchronize();
    
    //compute_sum();
    global_compute_sum<<<blocks, threads_per_block>>>( bin1, bin2, p_ci, p_sumi);
    cudaDeviceSynchronize();

  /***********************************************************************************************************/
  // INSERT RIGHT CUDA SYNCHRONIZATION AT END!
  /***********************************************************************************************************/
}

void ripple_carry_adder()
{
  int clast=0, cnext=0;

  for(int i = 0; i < bits; i++)
    {
      cnext = (bin1[i] & bin2[i]) | ((bin1[i] | bin2[i]) & clast);
      sumrca[i] = bin1[i] ^ bin2[i] ^ clast;
      clast = cnext;
    }
}

void check_cla_rca()
{
  for(int i = 0; i < bits; i++)
    {
      if( sumrca[i] != sumi[i] )
	{
	  printf("Check: Found sumrca[%d] = %d, not equal to sumi[%d] = %d - stopping check here!\n",
		 i, sumrca[i], i, sumi[i]);
	  printf("bin1[%d] = %d, bin2[%d]=%d, gi[%d]=%d, pi[%d]=%d, ci[%d]=%d, ci[%d]=%d\n",
		 i, bin1[i], i, bin2[i], i, gi[i], i, pi[i], i, ci[i], i-1, ci[i-1]);
	  return;
	}
    }
  printf("Check Complete: CLA and RCA are equal\n");
}

int main(int argc, char *argv[])
{
  int randomGenerateFlag = 1;
  int deterministic_seed = (1<<30) - 1;
  char* hexa=NULL;
  char* hexb=NULL;
  char* hexSum=NULL;
  char* int2str_result=NULL;
  //unsigned long long start_time=clock_now(); // dummy clock reads to init
  //unsigned long long end_time=clock_now();   // dummy clock reads to init

  if( nsupersupersections != block_size )
    {
      printf("Misconfigured CLA - nsupersupersections (%d) not equal to block_size (%d) \n",
	     nsupersupersections, block_size );
      return(-1);
    }
  
  if (argc == 2) {
    if (strcmp(argv[1], "-r") == 0)
      randomGenerateFlag = 1;
  }
  
  if (randomGenerateFlag == 0)
    {
      read_input();
    }
  else
    {
      srand( deterministic_seed );
      hex1 = generate_random_hex(input_size);
      hex2 = generate_random_hex(input_size);
    }
  
  hexa = prepend_non_sig_zero(hex1);
  hexb = prepend_non_sig_zero(hex2);
  hexa[digits] = '\0'; //double checking
  hexb[digits] = '\0';
  
  bin1 = gen_formated_binary_from_hex(hexa);
  bin2 = gen_formated_binary_from_hex(hexb);

  //allocate_cuda_memory();//allocate the memory for the cuda stuff
  	cudaMallocManaged (&p_gi, bits*sizeof(int));
	cudaMallocManaged (&p_pi, bits*sizeof(int));
	cudaMallocManaged (&p_ci, bits*sizeof(int));

	cudaMallocManaged (&p_ggj, ngroups*sizeof(int));
	cudaMallocManaged (&p_gpj, ngroups*sizeof(int));
	cudaMallocManaged (&p_gcj, ngroups*sizeof(int));

	cudaMallocManaged (&p_sgk, nsections*sizeof(int));
	cudaMallocManaged (&p_spk, nsections*sizeof(int));
	cudaMallocManaged (&p_sck, nsections*sizeof(int));

	cudaMallocManaged (&p_ssgl, nsupersections*sizeof(int));
	cudaMallocManaged (&p_sspl, nsupersections*sizeof(int));
	cudaMallocManaged (&p_sscl, nsupersections*sizeof(int));

	cudaMallocManaged (&p_sssgm, nsupersupersections*sizeof(int));
	cudaMallocManaged (&p_ssspm, nsupersupersections*sizeof(int));
	cudaMallocManaged (&p_ssscm, nsupersupersections*sizeof(int));

	cudaMallocManaged (&p_sumi, bits*sizeof(int));
	cudaMallocManaged (&bin1, bits*sizeof(int));
	cudaMallocManaged (&bin2, bits*sizeof(int));

/******************************************/
 // start_time = clock_now();
  cla(p_gi,p_pi,p_ci,   p_ggj,p_gpj,p_gcj,   p_sgk,p_spk,p_sck,   p_ssgl,p_sspl,p_sscl,   p_sssgm,p_ssspm,p_ssscm,   p_sumi,bin1,bin2);
  //end_time = clock_now();
/******************************************/
//  printf("CLA Completed in %llu cycles\n", (end_time - start_time));

  //start_time = clock_now();
  ripple_carry_adder();
  //end_time = clock_now();

  //printf("RCA Completed in %llu cycles\n", (end_time - start_time));

  check_cla_rca();

  if( verbose==1 )
    {
      int2str_result = int_to_string(sumi,bits);
      hexSum = revbinary_to_hex( int2str_result,bits);
    }

  // free inputs fields allocated in read_input or gen random calls
  free(int2str_result);
  free(hex1);
  free(hex2);
  
  // free bin conversion of hex inputs
  //free(bin1);
  //free(bin2);
  
  if( verbose==1 )
    {
      printf("Hex Input\n");
      printf("a   ");
      print_chararrayln(hexa);
      printf("b   ");
      print_chararrayln(hexb);
    }
  
  if ( verbose==1 )
    {
      printf("Hex Return\n");
      printf("sum =  ");
    }
  
  // free memory from prepend call
  free(hexa);
  free(hexb);

  if( verbose==1 )
    printf("%s\n",hexSum);
  
  free(hexSum);
  //free_cuda_memory();//free the cuda stuff
  	cudaFree (&p_gi);
	cudaFree (&p_pi);
	cudaFree (&p_ci);

	cudaFree (&p_ggj);
	cudaFree (&p_gpj);
	cudaFree (&p_gcj);

	cudaFree (&p_sgk);
	cudaFree (&p_spk);
	cudaFree (&p_sck);

	cudaFree (&p_ssgl);
	cudaFree (&p_sspl);
	cudaFree (&p_sscl);

	cudaFree (&p_sssgm);
	cudaFree (&p_ssspm);
	cudaFree (&p_ssscm);

	cudaFree (&p_sumi);
	cudaFree (&bin1);
	cudaFree (&bin2);
  return 0;
}
