#include <cstring>

namespace lib {

template <typename T> struct default_memory {
  using value_type = T;
  std::allocator<T> std_allocator;

  T *allocate(size_t size) { return std_allocator.allocate(size); }

  template <typename F> F *allocate(size_t size) {
    std::allocator<F> allocator;
    return allocator.allocate(size);
  }

  constexpr void deallocate(T *p, size_t n) { std_allocator.deallocate(p, n); }

  template <typename F> void deallocate(F *p, size_t n) {
    std::allocator<F> allocator;
    allocator.deallocate(p, n);
  }

  void memcpy(void *dst, const void *src, size_t numBytes) {
    std::memcpy(dst, src, numBytes);
  }

  template <typename F> void offload(F lambda) { lambda(); }
};

#ifdef SYCL_LANGUAGE_VERSION
template <typename T> struct sycl_memory {
  using value_type = T;
  using device_type = sycl::device;

  sycl::device device_;
  sycl::context context_;
  sycl::usm::alloc kind_;
  size_t alignment_;
  sycl::queue offload_queue_;

  sycl_memory(sycl::queue queue,
              sycl::usm::alloc kind = sycl::usm::alloc::shared,
              size_t alignment = 1)
      : kind_(kind), alignment_(alignment), offload_queue_(queue),
        device_(queue.get_device()), context_(queue.get_context()) {}

  T *allocate(size_t n) {
    return sycl::aligned_alloc<T>(alignment_, n, device_, context_, kind_);
  }

  template <typename F> F *allocate(size_t n) {
    return sycl::aligned_alloc<F>(alignment_, n, device_, context_, kind_);
  }

  void deallocate(T *p, std::size_t n) { sycl::free(p, context_); }

  template <typename F> void deallocate(F *p, std::size_t n) {
    sycl::free(p, context_);
  }

  void memcpy(void *dst, const void *src, size_t numBytes) {
    offload_queue_.memcpy(dst, src, numBytes).wait();
  }

  template <typename F> void offload(F lambda) {
    if (kind_ == sycl::usm::alloc::device) {
      offload_queue_.single_task(lambda).wait();
    } else {
      lambda();
    }
  }
};
#endif

} // namespace lib