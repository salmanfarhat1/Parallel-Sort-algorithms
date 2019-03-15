#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include<omp.h>

#include <x86intrin.h>

#define NBEXPERIMENTS   7

static long long unsigned int experiments [NBEXPERIMENTS] ;

/*
quick sort -- sequential, parallel --
*/

static unsigned int     N ;

typedef  int  *array_int ;

static array_int X ;

void init_array (array_int T)
{
  register int i ;

  for (i = 0 ; i < N ; i++)
  {
    T [i] = N - i ;
  }
}

void print_array (array_int T)
{
  register int i ;
  printf("\n");
  for (i = 0 ; i < N ; i++)
  {
    printf ("%d ", T[i]) ;
  }
  printf ("\n") ;
}

int is_sorted (array_int T)
{
  register int i ;

  for (i = 1 ; i < N ; i++)
  {
    /* test designed specifically for our usecase */
    if (T[i-1] +1  != T [i] )
    return 0 ;
  }
  return 1 ;
}

long long unsigned int average (long long unsigned int *exps)
{
  unsigned int i ;
  long long unsigned int s = 0 ;

  for (i = 2; i < (NBEXPERIMENTS-2); i++)
  {
    s = s + exps [i] ;
  }

  return s / (NBEXPERIMENTS-2) ;
}


static int compare (const void *x, const void *y)
{
  /* TODO: comparison function to be used by qsort()*/

  /* cast x and y to int* before comparing */
  if((*(int *)x) == (*(int *)y))
  return 0;
  else if((*(int *)x) > (*(int *)y))
  return 1;
  else
  return -1;

}

void sequential_qsort_sort (int *T, const int size)
{

  /* TODO: sequential sorting based on libc qsort() function */
  qsort(T , size , sizeof(int) ,compare );

  return ;
}

/*
Merge two sorted chunks of array T!
The two chunks are of size size
First chunck starts at T[0], second chunck starts at T[size]
*/
void merge (int *T, const int size)
{
  int *X = (int *) malloc (2 * size * sizeof(int)) ;

  int i = 0 ;
  int j = size ;
  int k = 0 ;

  while ((i < size) && (j < 2*size))
  {
    if (T[i] < T [j])
    {
      X [k] = T [i] ;
      i = i + 1 ;
    }
    else
    {
      X [k] = T [j] ;
      j = j + 1 ;
    }
    k = k + 1 ;
  }

  if (i < size)
  {
    for (; i < size; i++, k++)
    {
      X [k] = T [i] ;
    }
  }
  else
  {
    for (; j < 2*size; j++, k++)
    {
      X [k] = T [j] ;
    }
  }

  memcpy (T, X, 2*size*sizeof(int)) ;
  //print_array(X);
  //print_array(T);
  free (X) ;

  return ;
}
void mergeParallel (int *T, const int size)
{
  //printf("Calling me");
  //print_array(T);
  int *X = (int *) malloc (2 * size * sizeof(int)) ;

  int i = 0 ;
  int j = size ;
  int k = 0 ;
  int threadNo ;
  //  #pragma omp parallel for schedule(dynamic)
  for (; i < size && j < 2*size;)
  {

    if (T[i] < T [j])
    {
      X [k] = T [i] ;
      i = i + 1 ;
    }
    else
    {
      X [k] = T [j] ;
      j = j + 1 ;
    }
    k = k + 1 ;
  }

  if (i < size)
  {
    // for (; i < size; i++, k++)
    // {
    //   X [k] = T [i] ;
    // }
    #pragma omp parallel firstprivate(i, k)
    {
      for(i = i+threadNo ; i < size ; i +=omp_get_max_threads(), k = k+threadNo)
      {
        X[k] = T[i];
      }
    }
  }
  else
  {
    // for (; j < 2*size; j++, k++)
    // {
    //   X [k] = T [j] ;
    // }
    #pragma omp parallel firstprivate(j, k)
    {
      for(j = j+threadNo ; j < 2*size ; j +=omp_get_max_threads(), k = k+threadNo)
      {
        X[k] = T[j];
      }
    }
  }

  memcpy (T, X, 2*size*sizeof(int)) ;
  //print_array(X);
  //print_array(T);
  free (X) ;

  return ;
}



void parallel_qsort_sort (int *T, const int size)
{
  int l,numThreads;
  //omp_set_num_threads(4);
  numThreads = omp_get_max_threads();

  /* TODO: parallel sorting based on libc qsort() function +
  * sequential merging */
  int i , j , n=size/omp_get_max_threads() , threadNo ,swap_sort =0;

  #pragma omp parallel private(threadNo , i , j , swap_sort) shared(n)
  {
    threadNo = omp_get_thread_num();

    sequential_qsort_sort(&T[threadNo * n] , n); // threads will call in there regions sort function

    #pragma omp barrier
  }

  for(i = 2 ; i <= numThreads ; i *=2 ){ // merge sequential 2 then 4 , 8 etc ...
    for( l = 0 ; l < size ; l += n*i){ // regions
      if( l % i == 0){
        //printf("\nln is %d rk is :%d\n" ,l , i);
        merge(&T[l] , (i/2)*n ); // (i/2)*n because if chunk is 4 we need to sort 4 with 4, so 0 to 8 then 8 to 16
      }
    }
  }

}

void parallel_qsort_sort1 (int *T, const int size)
{
  int i , j , n=size/omp_get_max_threads() , threadNo ,swap_sort =0;

  #pragma omp parallel private(threadNo , i , j , swap_sort) shared(n)
  {
    threadNo = omp_get_thread_num();

    sequential_qsort_sort(&T[threadNo * n] , n); // threads will call in there regions sort function

    #pragma omp barrier
    //print_array(T);
    int range = n , k = 1; // range equal chunk size and k start with 1
    for( i = 2 ; i <= omp_get_max_threads() ; i *= 2 ) // parallel merge all 2 together then 4 together  etc ..
    {
      if(threadNo % i == 0) // if thread mod i ==0 then we need to merge
      {
        merge(&T[threadNo*n] , range * k); //( range * k )is region size  * k, suppose we have 2 numbers per region we need 2 then 4 then 8 etc .. if 4 per region we need 4 8 etc ..
      }
      k = k*2;
      #pragma omp barrier
    }

  }
}


int main (int argc, char **argv)
{
  unsigned long long int start, end, residu ;
  unsigned long long int av ;
  unsigned int exp ;

  if (argc != 2)
  {
    fprintf (stderr, "qsort N \n") ;
    exit (-1) ;
  }

  N = 1 << (atoi(argv[1])) ;

  X = (int *) malloc (N * sizeof(int)) ;

  printf("--> Sorting an array of size %u\n",N);

  start = _rdtsc () ;
  end   = _rdtsc () ;
  residu = end - start ;

  printf("sequential sorting ...\n");

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
  {
    init_array (X) ;

    start = _rdtsc () ;

    sequential_qsort_sort (X, N) ;

    end = _rdtsc () ;
    experiments [exp] = end - start ;

    if (! is_sorted (X)){
      fprintf(stderr, "ERROR: the array is not properly sorted\n") ;
      exit (-1) ;
    }
  }

  av = average (experiments) ;
  printf ("\n qsort serial\t\t %Ld cycles\n\n", av-residu) ;


  printf("parallel (seq merge) ...\n");


  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
  {
    init_array (X) ;

    start = _rdtsc () ;

    parallel_qsort_sort (X, N) ;

    end = _rdtsc () ;
    experiments [exp] = end - start ;

    if (! is_sorted (X))
    {
      fprintf(stderr, "ERROR: the array is not properly sorted\n") ;
      exit (-1) ;
    }
  }

  av = average (experiments) ;
  printf ("\n qsort parallel (seq merge) \t %Ld cycles\n\n", av-residu) ;

  printf("parallel ...\n");

  //------------------------------------------------------------------------------------------------------------------------------------
  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
  {
    init_array (X) ;

    start = _rdtsc () ;

    parallel_qsort_sort1 (X, N) ;

    end = _rdtsc () ;
    experiments [exp] = end - start ;

    if (! is_sorted (X))
    {
      fprintf(stderr, "ERROR: the array is not properly sorted\n") ;
      exit (-1) ;
    }
  }

  av = average (experiments) ;
  printf ("\n qsort parallel \t %Ld cycles\n\n", av-residu) ;

  //   print_array (X) ;


}
