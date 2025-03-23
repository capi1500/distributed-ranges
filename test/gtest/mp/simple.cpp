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
  dr::mp::distributed_vector<int> array(10, dist);
  fill(array, 7);

  if (rank == 0) {
    for (int i = 0; i <= 5; i++) {
      std::cout << i << ": " << *(array.begin() + i).local() << "\n";
    }
    for (int i = 4; i <= 5; i++) {
      *(array.begin() + i).local() = i - 45 + 1;
    }
    for (int i = 0; i <= 5; i++) {
      std::cout << i << ": " << *(array.begin() + i).local() << "\n";
    }
  } else {
    for (int i = 9; i >= 4; i--) {
      std::cout << i << ": " << *(array.begin() + i).local() << "\n";
    }
    for (int i = 4; i <= 5; i++) {
      *(array.begin() + i).local() = -(i - 45 + 1);
    }
    for (int i = 4; i <= 9; i++) {
      std::cout << i << ": " << *(array.begin() + i).local() << "\n";
    }
  }
  std::cout << "Begin halo\n";
  array.halo().exchange();
  std::cout << "End halo\n";
  for (int i = 4; i <= 5; i++) {
    std::cout << i << ": " << *(array.begin() + i).local() << "\n";
  }
}