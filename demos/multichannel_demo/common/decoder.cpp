// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "decoder.hpp"

#include "perf_timer.hpp"

#include <atomic>
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <thread>
#include <type_traits>
#include <vector>
#include <utility>

#ifdef USE_LIBVA
#include <dlfcn.h>

#include <tbb/concurrent_queue.h>
#include <tbb/concurrent_unordered_map.h>

// #define VA_USE_X11

#include <va/va.h>
#include <va/va_vpp.h>

#ifdef VA_USE_X11
#include <X11/Xlib.h>
#include <va/va_x11.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <va/va_drm.h>
#endif

namespace {

#if 0
#define JPEG_DEBUG(expr) do {(void)(expr);} while (false)
#else
#define JPEG_DEBUG(expr) (void)0
#endif

namespace details {
template<typename T>
struct ptr {
    T* p;

    operator T*&() {
        return p;
    }

    operator const T*&() const {
        return p;
    }
};

template<typename F>
struct SizedMarkerHandler {
    F func;
    unsigned char type = 0;

    template<typename T>
    SizedMarkerHandler(T&& f, unsigned char t):
        func(std::forward<T>(f)),
        type(t) {}

    SizedMarkerHandler(const SizedMarkerHandler&) = delete;
    SizedMarkerHandler(SizedMarkerHandler&&) = default;

    template<typename T>
    bool operator()(unsigned char t, ptr<const T>& data, size_t size) {
        if (t == type) {
            assert(size >= 2);
            auto len = static_cast<size_t>(256 * data.p[0] + data.p[1]);
            assert(len >= 2);
            assert(len <= size);
            auto new_ptr = data.p + len;
            data.p += 2;
            func(data, len - 2);
            data.p = std::max(data.p, new_ptr);
            return true;
        }
        return false;
    }
};

template<typename Handler, typename... Rest>
struct MarkerHandlers {
    Handler handler;
    MarkerHandlers<Rest...> next;

    template<typename Arg, typename... Args>
    MarkerHandlers(Arg&& arg, Args&&... args):
        handler(std::forward<Arg>(arg)),
        next(std::forward<Args>(args)...) {}

    template<typename T>
    void operator()(unsigned char type, ptr<const T>& data, size_t size) {
        if (!handler(type, data, size)) {
            next(type, data, size);
        }
    }
};

template<>
struct MarkerHandlers<void> {
    template<typename T>
    void operator()(unsigned char type, details::ptr<const T>&, size_t) {
        (void)type;
        // ignore
        JPEG_DEBUG(
            std::cout << "ignored marker " << std::hex
                      << static_cast<int>(type) << std::dec << std::endl);
    }
};
}  // namespace details

template<typename F>
auto make_sized_handler(unsigned char type, F&& handler)
->details::SizedMarkerHandler<typename std::remove_reference<F>::type> {
    return details::SizedMarkerHandler<typename std::remove_reference<F>::type>
            (std::forward<F>(handler), type);
}

template<typename Range, typename... Handlers>
void process_markers(const Range& range, Handlers&&... handlers) {
    details::MarkerHandlers<Handlers..., void> handler(
                std::forward<Handlers>(handlers)...);
    auto sz = range.size();
    auto start = range.data();
    auto end = start + sz - 1;
    for (auto current = start; current < end;) {
        auto marker = static_cast<unsigned char>(*current);
        bool updated = false;
        if (0xff == marker) {
            auto type = static_cast<unsigned char>(*(current + 1));
            if (0x00 != type && 0xff != type) {
                current += 2;
                auto i = static_cast<size_t>(current - start + 1);
                auto size = sz - i;
                details::ptr<const typename std::remove_reference<decltype(*current)>::type> ptr;
                ptr.p = current;
                handler(type, ptr, size);
                current = ptr.p;
                updated = true;
            }
        }
        if (!updated) {
            ++current;
        }
    }
}

struct DataReader {
    DataReader(const void* data, size_t size):
        begin(static_cast<const unsigned char*>(data)), end(begin + size) {}

    ~DataReader() {
        assert(empty());
    }

    template<typename T>
    T read() {
        static_assert(std::is_integral<T>::value, "Must be integral");
        auto sz = sizeof(T);
        assert(size() >= sz);
        auto ptr = begin;
        begin += sz;
        typename std::make_unsigned<T>::type ret = 0;
        for (size_t i = 0; i < sz; ++i) {
            ret = static_cast<decltype(ret)>(ret * 256);
            ret += ptr[i];
        }
        return static_cast<T>(ret);
    }

    void advance(size_t count) {
        assert(count <= size());
        begin += count;
    }

    size_t size() const {
        return static_cast<size_t>(end - begin);
    }

    bool empty() const {
        return 0 == size();
    }

    const void* data() const {
        return begin;
    }

private:
    const unsigned char* begin = nullptr;
    const unsigned char* end = nullptr;
};

struct Buff {
    const unsigned char* p = nullptr;
    size_t s = 0;

    size_t size() const {
        return s;
    }

    const unsigned char* data() const {
        return p;
    }
};

#ifdef VA_USE_X11
struct XDispDeleter {
    void operator()(Display* disp) const {
        assert(nullptr != disp);
        XCloseDisplay(disp);
    }
};
using XDisplayPtr = std::unique_ptr<Display, XDispDeleter>;
#else
struct fd_wrapper {
    explicit fd_wrapper(int f = -1):
        fd(f) {}

    ~fd_wrapper() {
        if (-1 != fd) {
            close(fd);
        }
    }

    fd_wrapper(const fd_wrapper&) = delete;
    fd_wrapper(fd_wrapper&& rhs) {
        std::swap(fd, rhs.fd);
    }

    fd_wrapper& operator=(const fd_wrapper&) = delete;
    fd_wrapper& operator=(fd_wrapper&& rhs) {
        if (this != &rhs) {
            std::swap(fd, rhs.fd);
        }
        return *this;
    }

    int get() const {
        return fd;
    }

private:
    int fd = -1;
};
#endif

struct VADeleter {
    void operator()(VADisplay disp) const {
        assert(nullptr != disp);
        vaTerminate(disp);
    }
};
using VAPtr = std::unique_ptr<std::remove_pointer<VADisplay>::type, VADeleter>;

void check_va(VAStatus status, const char* func) {
    if (VA_STATUS_SUCCESS != status) {
        std::stringstream ss;
        ss  << '\"' << func << "\" failed: " << vaErrorStr(status) <<
               " (" << status << ")" << std::endl;
        throw std::runtime_error(ss.str());
    }
}

#define CHECK_VA(A) check_va((A), #A)

template<typename F>
void map_va_buffer(VADisplay display, VABufferID buffer, F&& func) {
    assert(nullptr != display);
    assert(VA_INVALID_ID != buffer);

    void* data = nullptr;
    CHECK_VA(vaMapBuffer(display, buffer, &data));
    assert(nullptr != data);
    func(data);
    CHECK_VA(vaUnmapBuffer(display, buffer));
}

unsigned make_size_val(unsigned w, unsigned h) {
    assert(w <= std::numeric_limits<unsigned short>::max());
    assert(h <= std::numeric_limits<unsigned short>::max());
    return w | (h << 16);
}

}  // namespace

struct Decoder::HwContext {
    enum { InvalidId = VA_INVALID_ID };
    using clock = std::chrono::high_resolution_clock;

    const Decoder::Settings settings;

#ifdef VA_USE_X11
    XDisplayPtr x_display;
#else
    fd_wrapper dri_fd;
#endif
    VAPtr va_display;

    enum {
        PicParam,
        IqMat,
        HuffTable,
        SliceParam,
        SliceBuff,

        MaxBuffers
    };
    struct FreeSurfDesc {
        VASurfaceID decode_surface = InvalidId;
        VASurfaceID convert_surface = InvalidId;
    };
    using FreeSurfQueue = tbb::concurrent_bounded_queue<FreeSurfDesc>;

    struct Context {
        VAContextID decode_context = InvalidId;
        VAContextID convert_context = InvalidId;
        FreeSurfQueue available_surfaces;
    };

    tbb::concurrent_unordered_map<unsigned, Context> contexts;

    struct BusySurfDesc {
        VASurfaceID decode_surface = InvalidId;
        VASurfaceID convert_surface = InvalidId;
        callback_t callback;
        FreeSurfQueue* available_surfaces = nullptr;
        clock::time_point start_time = {};
    };
    tbb::concurrent_bounded_queue<BusySurfDesc> busy_surfaces;

    PerfTimer perf_timer_decode;

    std::thread wait_thread;

    explicit HwContext(const Decoder::Settings& s):
        settings(s),
        perf_timer_decode(s.collect_stats ? PerfTimer::DefaultIterationsCount :
                                            0) {
#ifdef VA_USE_X11
        x_display.reset(XOpenDisplay(nullptr));
        if (nullptr == x_display) {
            throw std::runtime_error("XOpenDisplay failed");
        }

        va_display.reset(vaGetDisplay(x_display.get()));
        if (nullptr == va_display) {
            throw std::runtime_error("vaGetDisplay failed");
        }
#else
        dri_fd = fd_wrapper(open("/dev/dri/renderD128", O_RDWR));
        if (-1 == dri_fd.get()) {
            throw std::runtime_error("Cannot open dri device");
        }

        va_display.reset(vaGetDisplayDRM(dri_fd.get()));
        if (nullptr == va_display) {
            throw std::runtime_error("vaGetDisplayDRM failed");
        }
#endif

        int major_version = 0;
        int minor_version = 0;
        CHECK_VA(vaInitialize(va_display.get(), &major_version,
                              &minor_version));

        wait_thread = std::thread([this]() {
            while (true) {
                BusySurfDesc desc = {};
                busy_surfaces.pop(desc);
                assert((InvalidId == desc.decode_surface) ==
                       (InvalidId == desc.convert_surface));
                if (InvalidId == desc.decode_surface) {
                    break;
                }
                assert(nullptr != desc.available_surfaces);
                CHECK_VA(vaSyncSurface(va_display.get(), desc.convert_surface));
                cv::Mat mat;
                VAImage image = {};
                CHECK_VA(vaDeriveImage(va_display.get(), desc.convert_surface, &image));

                const auto format = image.format.fourcc;
                if (format != VA_FOURCC_NV12 &&
                    format != VA_FOURCC_ARGB) {
                    std::stringstream ss;
                    ss << "unsupported va image format: 0x" << std::hex << format;
                    throw std::runtime_error(ss.str());
                }

                map_va_buffer(va_display.get(), image.buf, [&](void* data) {
                    auto ptr = static_cast<unsigned char*>(data);
                    switch (format) {
                    case VA_FOURCC_ARGB: {
                        assert(1 == image.num_planes);
                        cv::Mat temp_mat(image.height, image.width, CV_8UC4, ptr + image.offsets[0], image.pitches[0]);
                        cv::cvtColor(temp_mat, mat, cv::COLOR_BGRA2BGR);
                        break;
                    }
                    case VA_FOURCC_NV12: {
                        assert(2 == image.num_planes);
                        cv::Mat temp_mat1(image.height, image.width, CV_8UC1, ptr + image.offsets[0], image.pitches[0]);
                        cv::Mat temp_mat2(image.height / 2, image.width / 2, CV_8UC2, ptr + image.offsets[1], image.pitches[1]);
                        cv::cvtColorTwoPlane(temp_mat1, temp_mat2, mat, cv::COLOR_YUV2BGR_NV12);
                        break;
                    }

                    default:
                        assert(false);
                    }
                });

                CHECK_VA(vaDestroyImage(va_display.get(), image.image_id));

                if (perf_timer_decode.enabled()) {
                    auto start_time = desc.start_time;
                    auto end_time = clock::now();
                    auto duration = (end_time - start_time);
                    perf_timer_decode.addValue(duration);
                }

                desc.available_surfaces->push(FreeSurfDesc{
                                                  desc.decode_surface,
                                                  desc.convert_surface});
                desc.callback(std::move(mat));
            }
        });
    }

    Context createContext(unsigned width, unsigned height) {
        Context ret;
        VASurfaceAttrib decode_surf_attrib = {};
        decode_surf_attrib.type  = VASurfaceAttribPixelFormat;
        decode_surf_attrib.flags = VA_SURFACE_ATTRIB_SETTABLE;
        decode_surf_attrib.value.type = VAGenericValueTypeInteger;
        decode_surf_attrib.value.value.i = VA_FOURCC_422H;

        VASurfaceAttrib convert_surf_attrib = {};
        convert_surf_attrib.type  = VASurfaceAttribPixelFormat;
        convert_surf_attrib.flags = VA_SURFACE_ATTRIB_SETTABLE;
        convert_surf_attrib.value.type = VAGenericValueTypeInteger;
        convert_surf_attrib.value.value.i = VA_FOURCC_422H;

        std::vector<VASurfaceID> decode_surfaces;
        std::vector<VASurfaceID> convert_surfaces;
        decode_surfaces.resize(settings.num_buffers, InvalidId);
        convert_surfaces.resize(settings.num_buffers, InvalidId);
        CHECK_VA(vaCreateSurfaces(va_display.get(), VA_RT_FORMAT_YUV420,
                                  width, height,
                                  decode_surfaces.data(), settings.num_buffers,
                                  &decode_surf_attrib, 1));
        CHECK_VA(vaCreateSurfaces(va_display.get(), VA_RT_FORMAT_RGB32,
                                  settings.output_width, settings.output_height,
                                  convert_surfaces.data(), settings.num_buffers,
                                  &convert_surf_attrib, 0));

        for (unsigned i = 0; i < settings.num_buffers; ++i) {
            FreeSurfDesc freeSurf = {};
            freeSurf.decode_surface  = decode_surfaces[i];
            freeSurf.convert_surface = convert_surfaces[i];
            ret.available_surfaces.push(std::move(freeSurf));
        }

        auto createConfig = [this](
                            VAProfile profile, VAEntrypoint entrypoint) {
            VAConfigAttrib attrib = {};
            attrib.type = VAConfigAttribRTFormat;
            CHECK_VA(vaGetConfigAttributes(va_display.get(), profile, entrypoint,
                                           &attrib, 1));

            VAConfigID config = InvalidId;
            CHECK_VA(vaCreateConfig(va_display.get(), profile, entrypoint,
                                    &attrib, 1, &config));
            return config;
        };
        VAConfigID decode_config_id = createConfig(VAProfileJPEGBaseline,
                                                   VAEntrypointVLD);

        CHECK_VA(vaCreateContext(va_display.get(), decode_config_id,
                                 static_cast<int>(width),
                                 static_cast<int>(height),
                                 VA_PROGRESSIVE,
                                 decode_surfaces.data(),
                                 static_cast<int>(decode_surfaces.size()),
                                 &ret.decode_context));

        VAConfigID convert_config = createConfig(VAProfileNone,
                                                 VAEntrypointVideoProc);

        CHECK_VA(vaCreateContext(va_display.get(), convert_config,
                                 static_cast<int>(width),
                                 static_cast<int>(height),
                                 VA_PROGRESSIVE,
                                 convert_surfaces.data(),
                                 static_cast<int>(convert_surfaces.size()),
                                 &ret.convert_context));
        return ret;
    }

    ~HwContext() {
        busy_surfaces.push(BusySurfDesc{InvalidId, InvalidId, nullptr, {}});
        if (wait_thread.joinable()) {
            wait_thread.join();
        }
    }

    float getLatency() const {
        return perf_timer_decode.getValue();
    }

    void decodeImpl(const void* d, size_t s, VABufferID* buffers,
                    VAContextID context) {
        assert(InvalidId != context);
        VAHuffmanTableBufferJPEGBaseline huff_table = {};
        int restart_interval = 0;
        int image_width = 0;
        int image_height = 0;
        Buff buff;
        buff.p = static_cast<const unsigned char*>(d);
        buff.s = s;
        using Ptr = const unsigned char*;
        JPEG_DEBUG(std::cout << "========== jpeg begin ==========" << std::endl);
        process_markers(buff,
            make_sized_handler(0xDB, [&](Ptr data, size_t size) {
                JPEG_DEBUG(std::cout << "quant table size " << size << std::endl);
                DataReader reader(data, size);
                assert(buffers[Decoder::HwContext::IqMat] == InvalidId);
                VABufferID vabuff = InvalidId;
                CHECK_VA(vaCreateBuffer(
                             va_display.get(), context,
                             VAIQMatrixBufferType,
                             sizeof(VAIQMatrixBufferJPEGBaseline), 1,
                             nullptr, &vabuff));

                map_va_buffer(va_display.get(), vabuff, [&](void* buff_data) {
                    auto iq_table = static_cast<VAIQMatrixBufferJPEGBaseline*>(buff_data);
                    *iq_table = {};
                    while (!reader.empty()) {
                        int qt = reader.read<int8_t>();
                        auto qtnum = static_cast<int>(qt & 0xf);
                        auto precision = static_cast<int>((qt >> 4) & 0xf);
                        assert(0 == precision);
                        auto n = 64 * (precision + 1);
                        JPEG_DEBUG(std::cout << "  qt num " << qtnum << std::endl);
                        JPEG_DEBUG(std::cout << "  precision " << precision << std::endl);
                        JPEG_DEBUG(std::cout << "  n " << n);
                        JPEG_DEBUG(std::cout << std::hex << std::setw(2) << std::setfill('0'));
                        iq_table->load_quantiser_table[qtnum] = 1;
                        for (int i = 0; i < n; ++i) {
                            if (0 == (i % 16)) {
                                JPEG_DEBUG(std::cout << std::endl << "    ");
                            }
                            auto val = reader.read<uint8_t>();
                            iq_table->quantiser_table[qtnum][i] = val;
                            JPEG_DEBUG(
                                std::cout << std::setw(2)
                                          << static_cast<int>(val));
                        }
                        JPEG_DEBUG(
                            std::cout << std::dec << std::setw(0)
                                      << std::setfill(' ') << std::endl);
                    }
                });
                buffers[Decoder::HwContext::IqMat] = vabuff;
            }),
            make_sized_handler(0xC4, [&](Ptr data, size_t size) {
                JPEG_DEBUG(std::cout << "huff table size " << size << std::endl);
                DataReader reader(data, size);

                while (!reader.empty()) {
                    int ht = reader.read<int8_t>();
                    auto htnum = static_cast<int>(ht & 0xf);
                    auto type = static_cast<int>((ht >> 4) & 0x1);

                    huff_table.load_huffman_table[htnum] = 1;

                    auto& table = huff_table.huffman_table[htnum];

                    JPEG_DEBUG(std::cout << "  ht num " << htnum << std::endl);
                    JPEG_DEBUG(std::cout << "  ht type " << type << std::endl);
                    uint8_t* num_codes = (0 == type ? table.num_dc_codes :
                                                      table.num_ac_codes);

                    uint8_t* values = (0 == type ? table.dc_values :
                                                   table.ac_values);

                    int count = 0;
                    JPEG_DEBUG(std::cout << "  num ");
                    for (size_t i = 0; i < 16; ++i) {
                        int num = reader.read<uint8_t>();
                        JPEG_DEBUG(std::cout << num << " ");
                        count += num;
                        num_codes[i] = static_cast<uint8_t>(num);
                    }
                    JPEG_DEBUG(std::cout << std::endl);
                    JPEG_DEBUG(std::cout << "  sym count " << count);
                    JPEG_DEBUG(
                        std::cout << std::hex
                                  << std::setw(2) << std::setfill('0'));
                    for (int i = 0; i < count; ++i) {
                        if (0 == (i % 16)) {
                            JPEG_DEBUG(std::cout << std::endl << "    ");
                        }
                        auto val = reader.read<uint8_t>();
                        values[i] = val;
                        JPEG_DEBUG(
                            std::cout << std::setw(2)
                                      << static_cast<int>(val));
                    }
                    JPEG_DEBUG(
                        std::cout << std::dec << std::setw(0)
                                  << std::setfill(' ') << std::endl);
                }
            }),
            make_sized_handler(0xC0, [&](Ptr data, size_t size) {
                JPEG_DEBUG(std::cout << "img info size " << size << std::endl);
                DataReader reader(data, size);
                assert(buffers[Decoder::HwContext::PicParam] == InvalidId);
                VABufferID vabuff = InvalidId;
                CHECK_VA(vaCreateBuffer(
                             va_display.get(), context,
                             VAPictureParameterBufferType,
                             sizeof(VAPictureParameterBufferJPEGBaseline), 1,
                             nullptr, &vabuff));
                map_va_buffer(va_display.get(), vabuff, [&](void* buff_data) {
                    auto params = static_cast<VAPictureParameterBufferJPEGBaseline*>(buff_data);
                    *params = {};

                    int precision = reader.read<int8_t>();
                    (void)precision;
                    assert(8 == precision);
                    int height = reader.read<int16_t>();
                    int width  = reader.read<int16_t>();
                    image_width  = width;
                    image_height = height;
                    params->picture_width  = static_cast<uint16_t>(width);
                    params->picture_height = static_cast<uint16_t>(height);

                    JPEG_DEBUG(
                        std::cout << "  precision " << precision << std::endl);
                    JPEG_DEBUG(std::cout << "  height " << height << std::endl);
                    JPEG_DEBUG(std::cout << "  width "  << width  << std::endl);

                    auto componets = reader.read<int8_t>();
                    JPEG_DEBUG(
                        std::cout << "  components "
                                  << static_cast<int>(componets) << std::endl);
                    params->num_components = static_cast<uint8_t>(componets);
                    for (int i = 0; i < componets; ++i) {
                        int id = reader.read<int8_t>();
                        auto b = reader.read<int8_t>();
                        auto sampv = static_cast<int>(b & 0xf);
                        auto samph = static_cast<int>((b >> 4) & 0xf);
                        int quantn = reader.read<int8_t>();
                        JPEG_DEBUG(
                            std::cout << "    component " << id
                                      << " " << samph << "-" << sampv
                                      << " " << quantn << std::endl);

                        auto& comp = params->components[i];
                        comp.component_id = static_cast<uint8_t>(id);
                        comp.h_sampling_factor = static_cast<uint8_t>(samph);
                        comp.v_sampling_factor = static_cast<uint8_t>(sampv);
                        comp.quantiser_table_selector = static_cast<uint8_t>(quantn);
                    }
                });
                buffers[Decoder::HwContext::PicParam] = vabuff;
            }),
            make_sized_handler(0xDD, [&](Ptr data, size_t size) {
                JPEG_DEBUG(std::cout << "restart interval size " << size << std::endl);
                DataReader reader(data, size);
                int interval = reader.read<int16_t>();
                JPEG_DEBUG(std::cout << " interval " << interval << std::endl);
                restart_interval = interval;
            }),
            make_sized_handler(0xDA, [&](Ptr& data, size_t size) {
                JPEG_DEBUG(std::cout << "scan size " << size << std::endl);
                DataReader reader(data, size);

                int num_components = reader.read<int8_t>();

                using comp_type = std::remove_reference<decltype(VASliceParameterBufferJPEGBaseline().components[0])>::type;
                std::array<comp_type, 4> components = {};
                JPEG_DEBUG(std::cout << "  num components " << num_components << std::endl);
                for (int i = 0; i < num_components; ++i) {
                    int id = reader.read<int8_t>();
                    int table = reader.read<int8_t>();
                    auto actable = static_cast<int>(table & 0xf);
                    auto dctable = static_cast<int>((table >> 4) & 0xf);
                    JPEG_DEBUG(
                        std::cout << "    component " << id
                                  << "  ac "          << actable
                                  << "  dc "          << dctable
                                  << std::endl);
                    auto& comp = components[static_cast<size_t>(i)];
                    comp.component_selector = static_cast<uint8_t>(id);
                    comp.ac_table_selector = static_cast<uint8_t>(actable);
                    comp.dc_table_selector = static_cast<uint8_t>(dctable);
                }
                reader.advance(3);
                auto ptr = static_cast<const unsigned char*>(reader.data());
                auto begin = ptr;

                int scan_size = 0;
                auto print = [](unsigned char c) {
                    (void)c;
//                    JPEG_DEBUG(
//                        std::cout << std::setw(2) << std::hex
//                                  << static_cast<int>(c));
                };

                while (true) {
                    auto val = *ptr;
                    if (0xff == val) {
                        auto next = *(ptr + 1);
                        if (0x00 == next) {
                            print(val);
                            print(*(ptr + 2));
                            scan_size += 2;
                            ptr += 3;
                        } else if (next >= 0xD0 && next <= 0xD7) {
                            ptr += 2;
                        } else {
                            break;
                        }
                    } else {
                        print(val);
                        ++scan_size;
                        ++ptr;
                    }
                }
                (void)scan_size;
                JPEG_DEBUG(std::cout << "  scan data size " << scan_size << std::endl);
                auto data_size = (ptr - begin);
                JPEG_DEBUG(std::cout << "  computed data size " << data_size << std::endl);

                data = ptr;

                assert(buffers[Decoder::HwContext::SliceParam] == InvalidId);
                VABufferID vabuff1 = InvalidId;
                CHECK_VA(vaCreateBuffer(
                             va_display.get(), context,
                             VASliceParameterBufferType,
                             sizeof(VASliceParameterBufferJPEGBaseline), 1,
                             nullptr, &vabuff1));
                map_va_buffer(va_display.get(), vabuff1, [&](void* buff_data) {
                    auto slice = static_cast<VASliceParameterBufferJPEGBaseline*>(buff_data);
                    *slice = {};

                    slice->slice_data_size = static_cast<uint32_t>(data_size);
                    slice->slice_data_offset = 0;
                    slice->slice_data_flag = VA_SLICE_DATA_FLAG_ALL;
                    slice->restart_interval = static_cast<uint16_t>(restart_interval);

                    slice->num_components = static_cast<uint8_t>(num_components);
                    for (int i = 0; i < num_components; ++i) {
                        slice->components[i] = components[static_cast<size_t>(i)];
                    }
                    assert(0 != image_width);
                    assert(0 != image_height);
                    auto num_mcus = ((image_width + 7) / 8) *
                                    ((image_height + 7) / 8);
                    slice->num_mcus = static_cast<uint32_t>(num_mcus);
                });
                buffers[Decoder::HwContext::SliceParam] = vabuff1;

                VABufferID vabuff2 = InvalidId;
                CHECK_VA(vaCreateBuffer(
                             va_display.get(), context,
                             VASliceDataBufferType,
                             static_cast<unsigned>(data_size), 1,
                             nullptr, &vabuff2));
                map_va_buffer(va_display.get(), vabuff2, [&](void* buff_data) {
                    auto slice_data = static_cast<unsigned char*>(buff_data);
                    std::copy_n(begin, data_size, slice_data);
                });
                buffers[Decoder::HwContext::SliceBuff] = vabuff2;
            }));
        JPEG_DEBUG(std::cout << "========== jpeg end ==========" << std::endl);

        CHECK_VA(vaCreateBuffer(
                     va_display.get(), context,
                     VAHuffmanTableBufferType,
                     sizeof(VAHuffmanTableBufferJPEGBaseline), 1,
                     &huff_table, &buffers[Decoder::HwContext::HuffTable]));
        for (size_t i = 0; i < MaxBuffers; ++i) {
            assert(InvalidId != buffers[i]);
        }
    }

    void convertImpl(VABufferID& buff, VASurfaceID srcSurface,
                     VAConfigID context) {
        assert(InvalidId == buff);
        assert(InvalidId != context);

        VABufferID vabuff = InvalidId;
        CHECK_VA(vaCreateBuffer(
                     va_display.get(), context,
                     VAProcPipelineParameterBufferType,
                     sizeof(VAProcPipelineParameterBuffer), 1,
                     nullptr, &vabuff));
        map_va_buffer(va_display.get(), vabuff, [&](void* buff_data) {
            auto params = static_cast<VAProcPipelineParameterBuffer*>(buff_data);
            *params = {};
            params->surface = srcSurface;
            params->filter_flags = VA_FILTER_SCALING_DEFAULT | VA_FRAME_PICTURE;
        });
        buff = vabuff;
    }

    void decode(const void* data, size_t size, unsigned width,
                unsigned height, callback_t callback) {
        auto start_time = perf_timer_decode.enabled() ? clock::now() :
                                                        clock::time_point{};
        auto size_val = make_size_val(width, height);
        auto it = contexts.find(size_val);
        if (contexts.end() == it) {
            auto ctx = createContext(width, height);
            auto res = contexts.insert({size_val, std::move(ctx)});
            it = res.first;
        }
        Context& ctx = it->second;

        auto available_surfaces = &ctx.available_surfaces;

        std::array<VABufferID, MaxBuffers> buffers;
        std::fill_n(buffers.begin(), buffers.size(), InvalidId);

        decodeImpl(data, size, buffers.data(), ctx.decode_context);

        FreeSurfDesc surfaceDesc = {};
        available_surfaces->pop(surfaceDesc);
        VASurfaceID decode_surface = surfaceDesc.decode_surface;
        assert(InvalidId != decode_surface);

        CHECK_VA(vaBeginPicture(va_display.get(), ctx.decode_context, decode_surface));
        CHECK_VA(vaRenderPicture(va_display.get(), ctx.decode_context, buffers.data(), buffers.size()));
        CHECK_VA(vaEndPicture(va_display.get(), ctx.decode_context));
        for (auto buff_id : buffers) {
            if (InvalidId != buff_id) {
                CHECK_VA(vaDestroyBuffer(va_display.get(), buff_id));
            }
        }

        VASurfaceID convert_surface = surfaceDesc.convert_surface;
        assert(InvalidId != convert_surface);

        VABufferID convert_buffer = InvalidId;
        convertImpl(convert_buffer, decode_surface, ctx.convert_context);

        assert(InvalidId != convert_buffer);
        CHECK_VA(vaBeginPicture(va_display.get(), ctx.convert_context, convert_surface));
        CHECK_VA(vaRenderPicture(va_display.get(), ctx.convert_context, &convert_buffer, 1));
        CHECK_VA(vaEndPicture(va_display.get(), ctx.convert_context));

        CHECK_VA(vaDestroyBuffer(va_display.get(), convert_buffer));

        busy_surfaces.push(BusySurfDesc{decode_surface,
                                        convert_surface,
                                        std::move(callback),
                                        available_surfaces,
                                        start_time});
    }
};

#endif

Decoder::Decoder(const Settings& s):
    settings(s) {
    if (Mode::Hw == settings.mode) {
#ifdef USE_LIBVA
        hw_context.reset(new HwContext(settings));
#else
        throw std::logic_error("Hardware decoding is not supported");
#endif
    }
}

Decoder::~Decoder() {
}

Decoder::Stats Decoder::getStats() const {
#ifdef USE_LIBVA
    if (nullptr != hw_context) {
        return {hw_context->getLatency()};
    }
#endif
    return {};
}

#ifdef USE_LIBVA
void Decoder::decode_hw(const void* data, size_t size, unsigned width,
                        unsigned height, callback_t callback) {
    assert(nullptr != hw_context);
    hw_context->decode(data, size, width, height, std::move(callback));
}
#endif
