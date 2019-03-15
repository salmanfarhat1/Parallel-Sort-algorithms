#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include<omp.h>

#include <x86intrin.h>

#define NBEXPERIMENTS   7

static long long unsigned int experiments [NBEXPERIMENTS] ;

static   unsigned int N ;

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

void merge_sort (int *T, const int size)
{
  if (size==1)
  return;

  merge_sort(T,size/2);
  merge_sort(T+size/2,(size+1)/2);
  merge(T,size/2);
}


void parallel_merge_sort (int *T, const int size)
{

  if (size==1)
  return;
  #pragma omp task
  merge_sort(T,size/2);

  #pragma omp task
  merge_sort(T+size/2,(size+1)/2);

  #pragma omp taskwait
  merge(T,size/2);
}
void parallel_merge_sort_reduced_tasks (int *T, const int size)
{

  if (size==1)
  return;
  #pragma omp task
  {
    merge_sort(T,size/2);
  }
  merge_sort(T+size/2,(size+1)/2);

  #pragma omp taskwait
  merge(T,size/2);
}


int main (int argc, char **argv)
{
  unsigned long long int start, end, residu ;
  unsigned long long int av ;
  unsigned int exp ;

    if (argc != 2)
    {
      fprintf (stderr, "mergesort N \n") ;
      exit (-1) ;
    }

  N = 1 << (atoi(argv[1])) ;
  X = (int *) malloc (N * sizeof(int)) ;

  printf("--> Sorting an array of size %u\n",N);

  start = _rdtsc () ;
  end   = _rdtsc () ;
  residu = end - start ;

  // print_array (X) ;

  printf("sequential sorting ...\n");


    for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      init_array (X) ;

      start = _rdtsc () ;

               merge_sort (X, N) ;

      end = _rdtsc () ;
      experiments [exp] = end - start ;

      if (! is_sorted (X))
	{
            fprintf(stderr, "ERROR: the array is not properly sorted\n") ;
            exit (-1) ;
	}
    }

  av = average (experiments) ;
  printf ("\n merge sort serial\t\t %Ld cycles\n\n", av-residu) ;

  printf("parallel sorting ...\n");

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      init_array (X) ;

      start = _rdtsc () ;
      #pragma omp parallel
      {
        #pragma omp single
        parallel_merge_sort (X, N) ;
      }


      end = _rdtsc () ;
      experiments [exp] = end - start ;

      if (! is_sorted (X))
	{
            fprintf(stderr, "ERROR: the array is not properly sorted\n") ;
            exit (-1) ;
	}
    }

  av = average (experiments) ;
  printf ("\n merge sort parallel with tasks\t %Ld cycles\n\n", av-residu) ;

  printf("parallel sorting with reduced tasks ...\n");

    for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
      {
        init_array (X) ;

        start = _rdtsc () ;
        #pragma omp parallel
        {
          #pragma omp single
          parallel_merge_sort_reduced_tasks (X, N) ;
        }


        end = _rdtsc () ;
        experiments [exp] = end - start ;

        if (! is_sorted (X))
  	{
              fprintf(stderr, "ERROR: the array is not properly sorted\n") ;
              exit (-1) ;
  	}
      }

    av = average (experiments) ;
    printf ("\n merge sort parallel with reduced tasks\t %Ld cycles\n\n", av-residu) ;


}
