#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "mpi.h"
#include "utils.h"

void Jacobi(double ** u_previous, double ** u_current, int X_min, int X_max, int Y_min, int Y_max) {
	int i,j;
	for (i=X_min;i<X_max;i++)
		for (j=Y_min;j<Y_max;j++)
			u_current[i][j]=(u_previous[i-1][j]+u_previous[i+1][j]+u_previous[i][j-1]+u_previous[i][j+1])/4.0;
}

int main(int argc, char ** argv) {
    int rank,size;
    int global[2],local[2]; //global matrix dimensions and local matrix dimensions (2D-domain, 2D-subdomain)
    int global_padded[2];   //padded global matrix dimensions (if padding is not needed, global_padded=global)
    int grid[2];            //processor grid dimensions
    int padded[2] = {0,0};
    int i,j,t;
    int global_converged=0, converged=0; //flags for convergence, global and per process
    MPI_Datatype dummy;     //dummy datatype used to align user-defined datatypes in memory

    struct timeval totals,totalf,comps,compf,comms,commf;
    double ttotal=0,tcomp=0,tcomm=0,total_time,comp_time,comm_time;
    
    double ** U = malloc(1), ** u_current, ** u_previous, ** swap; 
    //Global matrix, local current and previous matrices, pointer to swap between current and previous
    

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    //----Read 2D-domain dimensions and process grid dimensions from stdin----//

    if (argc!=5) {
        fprintf(stderr,"Usage: mpirun .... ./exec X Y Px Py");
        exit(-1);
    }
    else {
        global[0]=atoi(argv[1]);
        global[1]=atoi(argv[2]);
        grid[0]=atoi(argv[3]);
        grid[1]=atoi(argv[4]);
    }

    //----Create 2D-cartesian communicator----//
    //----Usage of the cartesian communicator is optional----//

    MPI_Comm CART_COMM;         //CART_COMM: the new 2D-cartesian communicator
    int periods[2]={0,0};       //periods={0,0}: the 2D-grid is non-periodic
    int rank_grid[2];           //rank_grid: the position of each process on the new communicator
		
    MPI_Cart_create(MPI_COMM_WORLD,2,grid,periods,0,&CART_COMM);    //communicator creation
    MPI_Cart_coords(CART_COMM,rank,2,rank_grid);	            //rank mapping on the new communicator
    //printf("rank:%d with rank_grid[0]:%d and rank_grid[1]:%d\n",rank,rank_grid[0],rank_grid[1]);

    //----Compute local 2D-subdomain dimensions----//
    //----Test if the 2D-domain can be equally distributed to all processes----//
    //----If not, pad 2D-domain----//
    
    for (i=0;i<2;i++) {
        if (global[i]%grid[i]==0) {
            local[i]=global[i]/grid[i];
            global_padded[i]=global[i];
        }
        else {
            local[i]=(global[i]/grid[i])+1;
            global_padded[i]=local[i]*grid[i];
	    padded[i] = 1;
        }
    }

	

    //----Allocate global 2D-domain and initialize boundary values----//
    //----Rank 0 holds the global 2D-domain----//
	
    if (rank==0) {
	free(U);
        U=allocate2d(global_padded[0],global_padded[1]);   
        init2d(U,global[0],global[1]);
    }

    //----Allocate local 2D-subdomains u_current, u_previous----//
    //----Add a row/column on each size for ghost cells----//

    u_current = allocate2d(local[0] + 2, local[1] + 2);
    u_previous = allocate2d(local[0] + 2, local[1] + 2);
  
    //----Distribute global 2D-domain from rank 0 to all processes----//

    //----Appropriate datatypes are defined here----//
              
    //----Datatype definition for the 2D-subdomain on the global matrix----//

    //int MPI_Type_vector(int count, int blocklength, int stride,
    //        MPI_Datatype oldtype, MPI_Datatype *newtype);
	
    MPI_Datatype global_block;
    MPI_Type_vector(local[0],local[1],global_padded[1],MPI_DOUBLE,&dummy);
    MPI_Type_create_resized(dummy,0,sizeof(double),&global_block);
    MPI_Type_commit(&global_block);

    //----Datatype definition for the 2D-subdomain on the local matrix----//
    //----Note: this datatype assumes that the local matrix is extended
    //----      by 2 rows and columns to accomodate received data--------//
	
    MPI_Datatype local_block;
    MPI_Type_vector(local[0],local[1],local[1]+2,MPI_DOUBLE,&dummy);
    MPI_Type_create_resized(dummy,0,sizeof(double),&local_block);
    MPI_Type_commit(&local_block);

    //----Rank 0 scatters the global matrix----//
	
    int group_size = grid[0] * grid[1];
    int * sendcounts = malloc(group_size * sizeof(int));
    int * displs = malloc(group_size * sizeof(int));
 	

    if (rank == 0) {
    	for (i = 0; i < grid[0]; i++) {
		for (j = 0; j < grid[1]; j++) {
			displs[grid[1]*i + j] = (local[0]*global_padded[1]*i+ local[1]*j);
		        sendcounts[grid[1]*i + j] = 1;
		}
	}
    }
	
    MPI_Scatterv(&(U[0][0]), sendcounts, displs, global_block, &(u_previous[1][1]), 1, local_block, 0, MPI_COMM_WORLD);
    MPI_Scatterv(&(U[0][0]), sendcounts, displs, global_block, &(u_current[1][1]), 1, local_block, 0, MPI_COMM_WORLD);
    /* Make sure u_current and u_previous are
		                              both initialized */

    if (rank==0)
        free2d(U,global_padded[0],global_padded[1]);

    //----Define datatypes or allocate buffers for message passing----//

    //----Datatype definition for the 2D-subdomain on the global matrix----//
	
    MPI_Datatype column;
    MPI_Type_vector(local[0] + 2, 1, local[1] + 2 , MPI_DOUBLE, &dummy);
    MPI_Type_create_resized(dummy, 0, sizeof(double), &column);
    MPI_Type_commit(&column);

    MPI_Datatype row;
    MPI_Type_contiguous(local[1], MPI_DOUBLE, &row);
    MPI_Type_commit(&row);
    //----Find the 4 neighbors with which a process exchanges messages----//

    int north, south, east, west;

    MPI_Cart_shift(CART_COMM, 0, 1, &north, &south); // If <0 then there is no such neighbor
    MPI_Cart_shift(CART_COMM, 1, 1, &west, &east);   // Boundary processes and maybe in padding	
    
    //printf("---------------------------\n");
    //printf("rank:%d north:%d south:%d east:%d west:%d\n",rank,north,south,east,west);
	/*Make sure you handle non-existing
		neighbors appropriately*/

    //---Define the iteration ranges per process-----//
	
	/*Three types of ranges:
		-internal processes
		-boundary processes
		-boundary processes and padded global array
	*/
	
    int i_min,i_max,j_min,j_max,len;
	
	len = 4;
	i_min = 1;
	i_max = local[0]+1;
	j_min = 1;
	j_max = local[1]+1;
	

	if (north < 0) {
		len--;
		i_min++;
	}
	if (east < 0) {
		len--;
		if (padded[1] == 1)
			j_max -= 2;
		else
			j_max --;
	}
	if (south < 0) {
		len--;
		if (padded[0] == 1)
			i_max -= 2;
		else
			i_max--;
	}
	if (west < 0) {
		len--;
		j_min++;
	}

    //printf("rank:%d with imax:%d, imin:%d, jmin:%d, jmax:%d\n",rank,i_max,i_min,j_min,j_max);

    MPI_Request *requests_recv = malloc(len * sizeof(MPI_Request)),*requests_send = malloc(len * sizeof(MPI_Request));
    MPI_Status *status = malloc(len * sizeof(MPI_Status));

    gettimeofday(&totals,NULL);

    //----Computational core----//   
	
	#ifdef TEST_CONV
    for (t=0;t<T && !global_converged;t++) {
	#endif
	#ifndef TEST_CONV
	#undef T
	#define T 256
	for (t=0;t<T;t++) {
	#endif

		gettimeofday(&comms,NULL);
		len=0;
		
		// int MPI_Isend(void *buf, int count, MPI_Datatype datatype, int dest,
            	//						int tag, MPI_Comm comm, MPI_Request *request)		
		
		if (north >= 0) {
			MPI_Isend(&(u_previous[1][1]), 1, row, north, t, MPI_COMM_WORLD, requests_send+len);
			MPI_Irecv(&(u_previous[0][1]), 1, row, north, t, MPI_COMM_WORLD, requests_recv+len);
			len++;
		}		
		if (south >= 0) {
			MPI_Isend(&(u_previous[local[0]][1]), 1, row, south, t, MPI_COMM_WORLD, requests_send+len);
			MPI_Irecv(&(u_previous[local[0] + 1][1]), 1, row, south, t, MPI_COMM_WORLD, requests_recv+len);
			len++;
		}
		if (east >= 0) {
			MPI_Isend(&(u_previous[0][local[1]]), 1, column, east, t, MPI_COMM_WORLD, requests_send+len);
			MPI_Irecv(&(u_previous[0][local[1] + 1]), 1, column, east, t, MPI_COMM_WORLD, requests_recv+len);
			len++;
		}
		if (west >= 0) {
			MPI_Isend(&(u_previous[0][1]), 1, column, west, t, MPI_COMM_WORLD, requests_send+len);
			MPI_Irecv(&(u_previous[0][0]), 1, column, west, t, MPI_COMM_WORLD, requests_recv+len);
			len++;
		}
		MPI_Waitall(len, requests_send, status);
		MPI_Waitall(len, requests_recv, status);
		gettimeofday(&commf,NULL);

		tcomm+=(commf.tv_sec-comms.tv_sec)+(commf.tv_usec-comms.tv_usec)*0.000001;

		gettimeofday(&comps,NULL);
		Jacobi(u_previous,u_current,i_min,i_max,j_min,j_max);
		gettimeofday(&compf,NULL);
		tcomp+=(compf.tv_sec-comps.tv_sec)+(compf.tv_usec-comps.tv_usec)*0.000001;

		#ifdef TEST_CONV
        	if (t%C==0) {
			converged = converge(u_previous, u_current, i_min, i_max, j_min, j_max);
			MPI_Allreduce(&converged, &global_converged, 1, MPI_INT, MPI_PROD, MPI_COMM_WORLD);
		}		
		#endif
		
		swap=u_previous;
		u_previous=u_current;
		u_current=swap;
        
    }
    gettimeofday(&totalf,NULL);

    ttotal=(totalf.tv_sec-totals.tv_sec)+(totalf.tv_usec-totals.tv_usec) * 0.000001;

    MPI_Reduce(&ttotal,&total_time,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(&tcomp,&comp_time,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(&tcomm,&comm_time,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

    //----Rank 0 gathers local matrices back to the global matrix----//

    if (rank==0) {
    		U=allocate2d(global_padded[0],global_padded[1]);
    }

    MPI_Gatherv(&(u_previous[1][1]), 1, local_block, &(U[0][0]), sendcounts, displs, global_block, 0, MPI_COMM_WORLD);

    //----Printing results----//

    if (rank==0) {
		printf("Jacobi X %d Y %d Px %d Py %d Iter %d CommunicationTime %lf ComputationTime %lf TotalTime %lf midpoint %lf\n",
				global[0],global[1],grid[0],grid[1],t,comm_time,comp_time,total_time,U[global[0]/2][global[1]/2]);
	
		#ifdef PRINT_RESULTS
			char * s=malloc(50*sizeof(char));
		        sprintf(s,"resJacobiMPI_%dx%d_%dx%d",global[0],global[1],grid[0],grid[1]);
		    	fprint2d(s,U,global[0],global[1]);
		        free(s);
		#endif

    }

    MPI_Finalize();  
    return 0;
}


