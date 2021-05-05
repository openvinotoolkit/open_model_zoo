#include "vaapi_image_pool.h"
#include "vaapi_images.h"
#include "vaapi_context.h"

VaSurfacesPool::~VaSurfacesPool() {
    waitForCompletion();
}

VASurfaceID VaSurfacesPool::acquire(uint16_t width, uint16_t height, FourCC fourcc) {
    uint64_t key = calcKey(width, height, fourcc);
    std::unique_lock<std::mutex> lock(mtx);

    auto its = images.equal_range(key);

    for (auto it = its.first; it != its.second; ++it) {
        auto& foundElement = it->second;
        if (!foundElement.second) {
            foundElement.second = true;
            return foundElement.first;
        }
    }

    auto& elem = images.emplace(key,(Element(
        InferenceBackend::VaApiContext::createSurface(display,width,height,fourcc),
        true)))->second;

    return elem.first;
}

void VaSurfacesPool::release(const InferenceBackend::VaApiImage& img) {
    std::unique_lock<std::mutex> lock(mtx);
    auto its = images.equal_range(calcKey(img.width, img.height, img.format));

    for (auto it = its.first; it != its.second; ++it) {
        auto& foundElement = it->second;
        if(foundElement.first == img.va_surface_id) {
            foundElement.second = false;
            _free_image_condition_variable.notify_one();
            return;
        }
    }
    throw std::runtime_error("VaSurfacesPool: An attempt to release non-pooled surface is detected");
}

void VaSurfacesPool::waitForCompletion() {
    std::unique_lock<std::mutex> lock(mtx);
    for (auto& imagePair : images) {
        while(imagePair.second.second)
            _free_image_condition_variable.wait(lock);
    }
}
