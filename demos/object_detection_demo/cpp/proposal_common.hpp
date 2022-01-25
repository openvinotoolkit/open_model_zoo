#include <models/model_base.h>
#include <models/results.h>
#include <pipelines/metadata.h>
#include <utils/config_factory.h>
#include <utils/performance_metrics.hpp>
#include <utils/slog.hpp>

#include <chrono>
#include <queue>
#include <vector>

using Processor = ModelBase;

static inline void catcher() {
    std::exception_ptr ptr = std::current_exception();
    if (ptr) {
        try {
            std::rethrow_exception(ptr);
        } catch (const std::exception& exception) {
            slog::err << exception.what() << slog::endl;
        } catch (...) {
            slog::err << "Non exception object thrown" << slog::endl;
        }
    }
    std::abort();
}

class ColorPalette {
    // Move from main.cpp to common
};

struct LazyVideoWriter {
    void write(const cv::Mat&) {
        // Opens cv::VideoWriter at first write
    }
};

struct State {
    enum class StateEnum : bool {has_predictions, has_free_input};
    StateEnum state;
    virtual ~State() = default;
};

struct DetectionPredictions : public ResultBase, State {
    DetectionPredictions(int64_t frameId = -1, const std::shared_ptr<MetaData>& metaData = nullptr) :
        ResultBase(frameId, metaData) {}
    std::vector<DetectedObject> objects;
};

class SingleProcessorInferer {
    struct Input {
        cv::Mat mat;
        std::chrono::steady_clock::time_point time;
        unsigned idx;
    };
    std::queue<Input> mats;  // For Python there must be maxsize=nireq
    unsigned current_idx;
    unsigned nireq;
public:
    SingleProcessorInferer(std::shared_ptr<Processor> processor, const CnnConfig&) :
        current_idx{0} {}

    class Iter {
        SingleProcessorInferer& inferer;
    public:
        Iter(SingleProcessorInferer& inferer) : inferer{inferer} {}

        bool operator!=(const Iter&) {return true;}
        Iter& operator++() {
            // Returns end_sentinel if empty submit was done and everithing was infered
            return *this;
        }
        std::shared_ptr<State> operator*() {
            // Blocks and does the pipeline job.
            // Implementation of the job can be multithreaded.
            // Returns if it has predictions or SingleProcessorInferer can take a new mat.
            // Stops returning State::StateEnum::has_free_input after fist empty submit to SingleProcessorInferer
            // and throws if non empty submited after.
            return {};
        }
    };
    Iter begin() {return *this;}
    Iter end() {return *this;}

    void submit(cv::Mat&& mat, std::chrono::steady_clock::time_point start) {
        cv::Mat safe;
        if (nireq > 1) {
            // G-API team claims that for some tricky backends VideoCapture may not own memory thus they clone()
            safe = mat.clone();
        } else {
            safe = std::move(mat);
        }
        mats.push({safe, start, current_idx++});
    }

    PerformanceMetrics getInferenceMetircs() {return {};}
    PerformanceMetrics getPreprocessMetrics() {return {};}
    PerformanceMetrics getPostprocessMetrics() {return {};}
};
