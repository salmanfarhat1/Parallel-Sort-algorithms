#include <stdio.h>
#include<omp.h>

#include <x86intrin.h>

#define NBEXPERIMENTS    7
static long long unsigned int experiments [NBEXPERIMENTS] ;

/*
bubble sort -- sequential, parallel --
*/

static   unsigned int N ;

typedef  int  *array_int ;

/* the X array will be sorted  */

static   array_int X  ;

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

  for (i = 0 ; i < N ; i++)
  {
    printf ("%d ", T[i]) ;
  }
  printf ("\n") ;
}

/*
test if T is properly sorted
*/
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

void sequential_bubble_sort (int *T, const int size)
{
  int i, j ,swap_occur = 0 ,temp ;
  do
  {
    swap_occur = 0;
    for (j = 0; j < N-1; j++)
    if (T[j] > T[j+1])
    {

      temp = T[j];
      T[j] = T[j+1];
      T[j+1] = temp;
      swap_occur = 1;
    }
  }
  while(swap_occur == 1);

  return ;
}

void parallel_bubble_sort (int *T, const int size)
{
  /* TODO: parallel implementation of bubble sort */
  //omp_set_num_threads(4);
  int numThreads = omp_get_max_threads();
  int chunckSize = size/numThreads , i , allThreadsFinished = 1 ;
  if(chunckSize == 1){
    sequential_bubble_sort(T , size);
  }
  else // chuncksize is greater than 1
  {
    #pragma omp parallel private(i) shared(chunckSize ,allThreadsFinished)
    {
      int temp, swappedInner =0 ;

      while(allThreadsFinished != omp_get_max_threads() ){ // Exit when all thread didn't do swap, that means array is stable and well arranged
        //
        allThreadsFinished=0; // this is shared to check if all Threads are done
        swappedInner =0; // this is private
        #pragma omp for schedule(static , chunckSize) // each thread will take care of chuncksize
        for( i = 0 ; i < size ; i++)
        {
          if((i+1)%chunckSize != 0) // to not swap the last element of the chunck with the next chunck area
          {
            if(T[i] > T[i+1]) // swapping
            {
              temp = T[i];
              T[i] = T[i+1];
              T[i+1] = temp;
              swappedInner =1;
            }
          }
        } // here is a barrier

        #pragma omp barrier

        #pragma omp for schedule(static , 2)
        for(i = chunckSize - 1 ; i < size ; i +=chunckSize )
        {
          if(i != size-1) // if it is not the last element
          {
            if(T[i] > T[i+1]) // check if the last one in chunk is less than the first element of the next chunck if true swap
            {
              temp = T[i];
              T[i] = T[i+1];
              T[i+1] = temp;
              swappedInner = 1;
            }
          }
        } // barrier here

        if(swappedInner == 0){ // if threads didn't swap it will enter and increment allThreadsFinished by one
          #pragma omp critical // critical section
          allThreadsFinished += 1;
        }
        #pragma omp barrier

      }
    }
  }
  return ;
}


int main (int argc, char **argv)
{
  unsigned long long int start, end, residu ;
  unsigned long long int av ;
  unsigned int exp ;

  /* the program takes one parameter N which is the size of the array to
  be sorted. The array will have size 2^N */
  if (argc != 2)
  {
    fprintf (stderr, "bubble N \n") ;
    exit (-1) ;
  }

  N = 1 << (atoi(argv[1])) ;
  X = (int *) malloc (N * sizeof(int)) ;

  printf("--> Sorting an array of size %u\n",N);

  start = _rdtsc () ;
  end   = _rdtsc () ;
  residu = end - start ;


  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
  {
    init_array (X) ;

    start = _rdtsc () ;

    sequential_bubble_sort (X, N) ;

    end = _rdtsc () ;
    experiments [exp] = end - start ;

    /* verifying that X is properly sorted */
    if (! is_sorted (X))
    {
      fprintf(stderr, "ERROR: the array is not properly sorted\n") ;
      exit (-1) ;
    }
  }

  av = average (experiments) ;

  printf ("\n bubble serial \t\t\t %Ld cycles\n\n", av-residu) ;


  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
  {
    init_array (X) ;
    start = _rdtsc () ;

    parallel_bubble_sort (X, N) ;

    end = _rdtsc () ;
    experiments [exp] = end - start ;

    /* verifying that X is properly sorted */
    if (! is_sorted (X))
    {
      fprintf(stderr, "ERROR: the array is not properly sorted\n") ;
      exit (-1) ;
    }
  }

  av = average (experiments) ;
  printf ("\n bubble parallel \t %Ld cycles\n\n", av-residu) ;


  // print_array (X) ;


}
