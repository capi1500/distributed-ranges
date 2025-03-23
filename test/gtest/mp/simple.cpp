#include <dr/mp.hpp>
#include "mpi.h"

int main(int argc, char *argv[]) {
	MPI_Init(&argc, &argv);
	MPI_Comm comm = MPI_COMM_WORLD;
	int rank;
	int size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

#ifdef SYCL_LANGUAGE_VERSION
	sycl::queue q = dr::mp::select_queue();
	fmt::print("Running on sycl device: {}, memory: device\n", q.get_device().get_info<sycl::info::device::name>());
	dr::mp::init(q, sycl::usm::alloc::device);
#else
	assert(false)
#endif

	auto dist = dr::mp::distribution().halo(1);
	dr::mp::distributed_vector<int> array(100, dist);
	std::cout << "Begin halo\n";
	array.halo().exchange();
	std::cout << "End halo\n";

//	cudaMemcpyPeer();
}