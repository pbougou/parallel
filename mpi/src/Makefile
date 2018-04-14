CC=gcc
CFLAGS=-O3 -lm -Wall -Wno-unused-variable -Wno-unused-result -Wno-comment
RES=-DPRINT_RESULTS
CONV=-DTEST_CONV

all: jacobi_par jacobi256 seidelsor_par seidelsor256 redblacksor_par redblacksor256

jacobi_par: Jacobi_par.c utils.c
	mpicc $(CFLAGS) $(RES) $(CONV) Jacobi_par.c utils.c -o jacobi_par

jacobi256: Jacobi_par.c utils.c
	mpicc $(CFLAGS) Jacobi_par.c utils.c -o jacobi256

seidelsor_par: seidelsor_par.c utils.c
	mpicc $(CFLAGS) $(RES) $(CONV) seidelsor_par.c utils.c -o seidelsor_par

seidelsor256: seidelsor_par.c utils.c
	mpicc $(CFLAGS) seidelsor_par.c utils.c -o seidelsor256

redblacksor_par: redblacksor_par.c utils.c
	mpicc $(CFLAGS) $(RES) $(CONV) redblacksor_par.c utils.c -o redblacksor_par

redblacksor256: redblacksor_par.c utils.c
	mpicc $(CFLAGS) redblacksor_par.c utils.c -o redblacksor256


clean:
	rm jacobi_par jacobi256 seidelsor_par seidelsor256 redblacksor_par redblacksor256

