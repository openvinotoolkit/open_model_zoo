#pragma once 

#include "utils/uni_image_defs.h"
#include <unordered_map>
#include <future>
#include "fourcc.h"


namespace InferenceBackend {
  class VaApiContext;
  class VaApiImage;
}

class VaSurfacesPool {
  public:
    VASurfaceID acquire(uint16_t width, uint16_t height, FourCC fourcc);
    void release(const InferenceBackend::VaApiImage& img);
    void waitForCompletion();
    VaSurfacesPool() : display(nullptr) {}
    VaSurfacesPool(VADisplay display) : display(display) {}
    ~VaSurfacesPool();   
  private:
    using Element = std::pair<VASurfaceID, bool>; // second is true if image is in use
    std::unordered_multimap<uint64_t, Element> images;
    std::condition_variable _free_image_condition_variable;
    std::mutex mtx;

    uint64_t calcKey(uint16_t width, uint16_t height, FourCC fourcc) {
        return static_cast<uint64_t>(fourcc) |
            ((static_cast<uint64_t>(width) & 0xFFFF)<<32) | ((static_cast<uint64_t>(height) & 0xFFFF)<<48);
    }

    VADisplay display;
};
