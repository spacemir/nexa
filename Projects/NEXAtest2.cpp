// NEXAtest2.cpp : Defines the entry point for the console application.
//

#include <iostream>
//#include "stdafx.h"
#include "NetworkDemo0.h"
#include "NetworkMNIST.h"
#include <mpi.h>
#include "NetworkTemporal3BCPNN.h"

int main(int argc, char* argv[])
{
	int mpiRank, mpiSize;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
	
	NetworkTemporal3BCPNN network;
	network.Run();
	
	/* 
	NetworkTemporalBCPNN*/


	//NetworkDemo0 networkdemo;
	//networkdemo.Run();
	//if (mpiRank = 0){
	//	cout << "networkdemo0 object created and ran" << endl;
	//}
	//

	//NetworkMNIST2 mnist;
	//mnist.Run();

	MPI_Finalize();
	return 0;
}

