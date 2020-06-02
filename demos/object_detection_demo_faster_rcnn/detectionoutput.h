// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cfloat>
#include <vector>
#include <cmath>
#include <string>
#include <utility>
#include <algorithm>

#include <inference_engine.hpp>

using namespace InferenceEngine;
using InferenceEngine::details::InferenceEngineException;

template <typename T>
static bool SortScorePairDescend(const std::pair<float, T>& pair1,
                                 const std::pair<float, T>& pair2) {
    return pair1.first > pair2.first;
}

/**
 * @class DetectionOutputPostProcessor
 * @brief This class is almost a copy of MKLDNN extension implementation DetectionOutputImpl.
 * In our demo we use it as a post-processing class for Faster-RCNN networks
 */
class DetectionOutputPostProcessor {
public:
    explicit DetectionOutputPostProcessor(const SizeVector &image_dims,
                                          const SizeVector &loc_dims,
                                          const SizeVector &conf_dims,
                                          const SizeVector &prior_dims) {
        try {
            IE_ASSERT(4 == image_dims.size());

            _image_height = image_dims[2];
            _image_width = image_dims[3];

            IE_ASSERT(2 <=  prior_dims.size());
            int priors_size = prior_dims[prior_dims.size() - 2] * prior_dims[prior_dims.size() - 1];

            IE_ASSERT(2 <= loc_dims.size());
            int loc_size = loc_dims[loc_dims.size() - 2]*loc_dims[loc_dims.size() - 1];

            IE_ASSERT(2 <= conf_dims.size());
            size_t conf_size = conf_dims[0]*conf_dims[1];

            _num_priors = static_cast<int>(priors_size / _prior_size);

            // num_classes guessed from the output dims
            if (loc_size % (_num_priors * 4) != 0) {
                throw std::runtime_error("Can't guess number of classes. Something's wrong with output layers dims");
            }

            _num_classes = loc_size / (_num_priors * 4);
            _num_loc_classes = _num_classes;

            if (_num_priors * _num_loc_classes * 4 != loc_size)
                THROW_IE_EXCEPTION << "Number of priors must match number of location predictions.";

            if (_num_priors * _num_classes != static_cast<int>(conf_size))
                THROW_IE_EXCEPTION << "Number of priors must match number of confidence predictions.";

            SizeVector bboxes_size{conf_size,
                                                    static_cast<size_t>(_num_classes),
                                                    static_cast<size_t>(_num_priors),
                                                    4};
            _decoded_bboxes = make_shared_blob<float>({Precision::FP32, bboxes_size, NCHW});
            _decoded_bboxes->allocate();

            SizeVector buf_size{conf_size,
                                                 static_cast<size_t>(_num_classes),
                                                 static_cast<size_t>(_num_priors)};
            _buffer = make_shared_blob<int>({Precision::I32, buf_size, {buf_size, {0, 1, 2}}});
            _buffer->allocate();

            SizeVector indices_size{conf_size,
                                                     static_cast<size_t>(_num_classes),
                                                     static_cast<size_t>(_num_priors)};
            _indices = make_shared_blob<int>(
                    {Precision::I32, indices_size, {indices_size, {0, 1, 2}}});
            _indices->allocate();

            SizeVector detections_size{conf_size * static_cast<size_t>(_num_classes)};
            _detections_count = make_shared_blob<int>({Precision::I32, detections_size, C});
            _detections_count->allocate();

            SizeVector conf_size1 = { conf_size, 1 };
            _reordered_conf = make_shared_blob<float>({Precision::FP32, conf_size1, ANY});
            _reordered_conf->allocate();

            SizeVector decoded_bboxes_size{conf_size,
                                                            static_cast<size_t>(_num_priors),
                                                            static_cast<size_t>(_num_classes)};
            _bbox_sizes = make_shared_blob<float>(
                    {Precision::FP32, decoded_bboxes_size, {decoded_bboxes_size, {0, 1, 2}}});
            _bbox_sizes->allocate();

            SizeVector num_priors_actual_size{conf_size};
            _num_priors_actual = make_shared_blob<int>({Precision::I32, num_priors_actual_size, C});
            _num_priors_actual->allocate();
        } catch (const InferenceEngineException& ex) {
            throw std::logic_error(std::string("Can't create detection output: ") + ex.what());
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept {
        LockedMemory<void> outputMapped = as<MemoryBlob>(outputs[0])->wmap();
        float *dst_data = outputMapped;

        LockedMemory<const void> idxLocationMapped = as<MemoryBlob>(inputs[idx_location])->rmap();
        const float *loc_data   = idxLocationMapped.as<float*>();
        LockedMemory<const void> idxConfidenceMapped = as<MemoryBlob>(inputs[idx_confidence])->rmap();
        const float *conf_data  = idxConfidenceMapped.as<float*>();
        LockedMemory<const void> idxPriorsMapped = as<MemoryBlob>(inputs[idx_priors])->rmap();
        const float *prior_data = idxPriorsMapped.as<float*>();

        const int N = 1;  // TODO: Support batch

        LockedMemory<const void> decodedBboxesMapped = as<MemoryBlob>(_decoded_bboxes)->rmap();
        float *decoded_bboxes_data = decodedBboxesMapped.as<float*>();
        LockedMemory<const void> reorderedConfMapped = as<MemoryBlob>(_reordered_conf)->rmap();
        float *reordered_conf_data = reorderedConfMapped.as<float*>();
        LockedMemory<const void> bboxSizesMapped = as<MemoryBlob>(_bbox_sizes)->rmap();
        float *bbox_sizes_data     = bboxSizesMapped.as<float*>();
        LockedMemory<const void> detectionsCountMapped = as<MemoryBlob>(_detections_count)->rmap();
        int *detections_data       = detectionsCountMapped.as<int*>();
        LockedMemory<const void> bufferMapped = as<MemoryBlob>(_buffer)->rmap();
        int *buffer_data           = bufferMapped.as<int*>();
        LockedMemory<const void> indicesMapped = as<MemoryBlob>(_indices)->rmap();
        int *indices_data          = indicesMapped.as<int*>();
        LockedMemory<const void> numPriorsActualMapped = as<MemoryBlob>(_num_priors_actual)->rmap();
        int *num_priors_actual     = numPriorsActualMapped.as<int*>();

        const float *prior_variances = prior_data + _num_priors*_prior_size;
        const float *ppriors = prior_data;

        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < _num_loc_classes; ++c) {
                if (c == _background_label_id) {
                    continue;
                }

                const float *ploc = loc_data + n*4*_num_loc_classes*_num_priors + c*4;
                float *pboxes = decoded_bboxes_data + n*4*_num_loc_classes*_num_priors + c*4*_num_priors;
                float *psizes = bbox_sizes_data + n*_num_loc_classes*_num_priors + c*_num_priors;
                decodeBBoxes(ppriors, ploc, prior_variances, pboxes, psizes, num_priors_actual, n);
            }
        }

        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < _num_classes; ++c) {
                for (int p = 0; p < _num_priors; ++p) {
                    reordered_conf_data[n*_num_priors*_num_classes + c*_num_priors + p] = conf_data[n*_num_priors*_num_classes + p*_num_classes + c];
                }
            }
        }

        memset(detections_data, 0, N*_num_classes*sizeof(int));

        for (int n = 0; n < N; ++n) {
            int detections_total = 0;

            for (int c = 0; c < _num_classes; ++c) {
                if (c == _background_label_id) {
                    // Ignore background class.
                    continue;
                }

                int *pindices    = indices_data + n*_num_classes*_num_priors + c*_num_priors;
                int *pbuffer     = buffer_data + c*_num_priors;
                int *pdetections = detections_data + n*_num_classes + c;

                const float *pconf = reordered_conf_data + n*_num_classes*_num_priors + c*_num_priors;
                const float *pboxes = decoded_bboxes_data + n*4*_num_classes*_num_priors + c*4*_num_priors;
                const float *psizes = bbox_sizes_data + n*_num_classes*_num_priors + c*_num_priors;

                nms(pconf, pboxes, psizes, pbuffer, pindices, *pdetections, num_priors_actual[n]);
            }

            for (int c = 0; c < _num_classes; ++c) {
                detections_total += detections_data[n*_num_classes + c];
            }

            if (_keep_top_k > -1 && detections_total > _keep_top_k) {
                std::vector<std::pair<float, std::pair<int, int>>> conf_index_class_map;

                for (int c = 0; c < _num_classes; ++c) {
                    int detections = detections_data[n*_num_classes + c];
                    int *pindices = indices_data + n*_num_classes*_num_priors + c*_num_priors;
                    float *pconf  = reordered_conf_data + n*_num_classes*_num_priors + c*_num_priors;

                    for (int i = 0; i < detections; ++i) {
                        int idx = pindices[i];
                        conf_index_class_map.push_back(std::make_pair(pconf[idx], std::make_pair(c, idx)));
                    }
                }

                std::sort(conf_index_class_map.begin(), conf_index_class_map.end(),
                          SortScorePairDescend<std::pair<int, int>>);
                conf_index_class_map.resize(_keep_top_k);

                // Store the new indices.
                memset(detections_data + n*_num_classes, 0, _num_classes * sizeof(int));

                for (size_t j = 0; j < conf_index_class_map.size(); ++j) {
                    int label = conf_index_class_map[j].second.first;
                    int idx = conf_index_class_map[j].second.second;
                    int *pindices = indices_data + n * _num_classes * _num_priors + label * _num_priors;
                    pindices[detections_data[n*_num_classes + label]] = idx;
                    detections_data[n*_num_classes + label]++;
                }
            }
        }

        const int DETECTION_SIZE = outputs[0]->getTensorDesc().getDims()[3];
        if (DETECTION_SIZE != 7) {
            return NOT_IMPLEMENTED;
        }

        auto dst_data_size = N * _keep_top_k * DETECTION_SIZE * sizeof(float);

        if (dst_data_size > outputs[0]->byteSize()) {
            return OUT_OF_BOUNDS;
        }

        memset(dst_data, 0, dst_data_size);

        int count = 0;
        for (int n = 0; n < N; ++n) {
            const float *pconf   = reordered_conf_data + n * _num_priors * _num_classes;
            const float *pboxes  = decoded_bboxes_data + n*_num_priors*4*_num_loc_classes;
            const int *pindices  = indices_data + n*_num_classes*_num_priors;

            for (int c = 0; c < _num_classes; ++c) {
                for (int i = 0; i < detections_data[n*_num_classes + c]; ++i) {
                    int idx = pindices[c*_num_priors + i];

                    dst_data[count * DETECTION_SIZE + 0] = static_cast<float>(n);
                    dst_data[count * DETECTION_SIZE + 1] = static_cast<float>(c);
                    dst_data[count * DETECTION_SIZE + 2] = pconf[c*_num_priors + idx];

                    float xmin = pboxes[c*4*_num_priors + idx*4 + 0];
                    float ymin = pboxes[c*4*_num_priors + idx*4 + 1];
                    float xmax = pboxes[c*4*_num_priors + idx*4 + 2];
                    float ymax = pboxes[c*4*_num_priors + idx*4 + 3];

                    dst_data[count * DETECTION_SIZE + 3] = xmin;
                    dst_data[count * DETECTION_SIZE + 4] = ymin;
                    dst_data[count * DETECTION_SIZE + 5] = xmax;
                    dst_data[count * DETECTION_SIZE + 6] = ymax;

                    ++count;
                }
            }
        }

        if (count < N*_keep_top_k) {
            // marker at end of boxes list
            dst_data[count * DETECTION_SIZE + 0] = -1;
        }

        return OK;
    }

private:
    const int idx_location = 0;
    const int idx_confidence = 1;
    const int idx_priors = 2;


    int _num_classes = 0;
    const int _background_label_id = 0;
    const int _top_k = 400;
    const int _keep_top_k = 200;

    int _image_width = 0;
    int _image_height = 0;
    const int _prior_size = 5;
    const int _offset = 1;

    const float _nms_threshold = 0.3f;
    const float _confidence_threshold = -FLT_MAX;

    int _num_loc_classes = 0;
    int _num_priors = 0;

    void decodeBBoxes(const float *prior_data, const float *loc_data, const float *variance_data,
                      float *decoded_bboxes, float *decoded_bbox_sizes, int* num_priors_actual, int n);

    void nms(const float *conf_data, const float *bboxes, const float *sizes,
             int *buffer, int *indices, int &detections, int num_priors_actual);

    Blob::Ptr _decoded_bboxes;
    Blob::Ptr _buffer;
    Blob::Ptr _indices;
    Blob::Ptr _detections_count;
    Blob::Ptr _reordered_conf;
    Blob::Ptr _bbox_sizes;
    Blob::Ptr _num_priors_actual;
};

struct ConfidenceComparator {
    explicit ConfidenceComparator(const float* conf_data) : _conf_data(conf_data) {}

    bool operator()(int idx1, int idx2) {
        if (_conf_data[idx1] > _conf_data[idx2]) return true;
        if (_conf_data[idx1] < _conf_data[idx2]) return false;
        return idx1 < idx2;
    }

    const float* _conf_data;
};

static inline float JaccardOverlap(const float *decoded_bbox,
                                   const float *bbox_sizes,
                                   const int idx1,
                                   const int idx2) {
    float xmin1 = decoded_bbox[idx1*4 + 0];
    float ymin1 = decoded_bbox[idx1*4 + 1];
    float xmax1 = decoded_bbox[idx1*4 + 2];
    float ymax1 = decoded_bbox[idx1*4 + 3];

    float xmin2 = decoded_bbox[idx2*4 + 0];
    float ymin2 = decoded_bbox[idx2*4 + 1];
    float ymax2 = decoded_bbox[idx2*4 + 3];
    float xmax2 = decoded_bbox[idx2*4 + 2];

    if (xmin2 > xmax1 || xmax2 < xmin1 || ymin2 > ymax1 || ymax2 < ymin1) {
        return 0.0f;
    }

    float intersect_xmin = std::max(xmin1, xmin2);
    float intersect_ymin = std::max(ymin1, ymin2);
    float intersect_xmax = std::min(xmax1, xmax2);
    float intersect_ymax = std::min(ymax1, ymax2);

    float intersect_width  = intersect_xmax - intersect_xmin;
    float intersect_height = intersect_ymax - intersect_ymin;

    if (intersect_width <= 0 || intersect_height <= 0) {
        return 0.0f;
    }

    float intersect_size = intersect_width * intersect_height;
    float bbox1_size = bbox_sizes[idx1];
    float bbox2_size = bbox_sizes[idx2];

    return intersect_size / (bbox1_size + bbox2_size - intersect_size);
}

void DetectionOutputPostProcessor::decodeBBoxes(const float *prior_data,
                                   const float *loc_data,
                                   const float *variance_data,
                                   float *decoded_bboxes,
                                   float *decoded_bbox_sizes,
                                   int* num_priors_actual,
                                   int n) {
    num_priors_actual[n] = _num_priors;

    for (int num = 0; num < _num_priors; ++num) {
        float batch_id = prior_data[num * _prior_size + 0];
        if (batch_id == -1.f) {
            num_priors_actual[n] = num;
            break;
        }
    }

    for (int p = 0; p < num_priors_actual[n]; ++p) {
        float prior_xmin = prior_data[p*_prior_size + 0 + _offset];
        float prior_ymin = prior_data[p*_prior_size + 1 + _offset];
        float prior_xmax = prior_data[p*_prior_size + 2 + _offset];
        float prior_ymax = prior_data[p*_prior_size + 3 + _offset];

        float loc_xmin = loc_data[4*p*_num_loc_classes + 0];
        float loc_ymin = loc_data[4*p*_num_loc_classes + 1];
        float loc_xmax = loc_data[4*p*_num_loc_classes + 2];
        float loc_ymax = loc_data[4*p*_num_loc_classes + 3];

        prior_xmin /= _image_width;
        prior_ymin /= _image_height;
        prior_xmax /= _image_width;
        prior_ymax /= _image_height;

        float prior_width    =  prior_xmax - prior_xmin;
        float prior_height   =  prior_ymax - prior_ymin;
        float prior_center_x = (prior_xmin + prior_xmax) / 2.0f;
        float prior_center_y = (prior_ymin + prior_ymax) / 2.0f;

        float decode_bbox_center_x, decode_bbox_center_y;
        float decode_bbox_width, decode_bbox_height;

        // variance is encoded in target, we simply need to restore the offset predictions.
        decode_bbox_center_x = loc_xmin * prior_width  + prior_center_x;
        decode_bbox_center_y = loc_ymin * prior_height + prior_center_y;
        decode_bbox_width  = std::exp(loc_xmax) * prior_width;
        decode_bbox_height = std::exp(loc_ymax) * prior_height;

        float new_xmin = decode_bbox_center_x - decode_bbox_width  / 2.0f;
        float new_ymin = decode_bbox_center_y - decode_bbox_height / 2.0f;
        float new_xmax = decode_bbox_center_x + decode_bbox_width  / 2.0f;
        float new_ymax = decode_bbox_center_y + decode_bbox_height / 2.0f;

        decoded_bboxes[p*4 + 0] = new_xmin;
        decoded_bboxes[p*4 + 1] = new_ymin;
        decoded_bboxes[p*4 + 2] = new_xmax;
        decoded_bboxes[p*4 + 3] = new_ymax;

        decoded_bbox_sizes[p] = (new_xmax - new_xmin) * (new_ymax - new_ymin);
    }
}

void DetectionOutputPostProcessor::nms(const float* conf_data,
                          const float* bboxes,
                          const float* sizes,
                          int* buffer,
                          int* indices,
                          int& detections,
                          int num_priors_actual) {
    int count = 0;
    for (int i = 0; i < num_priors_actual; ++i) {
        if (conf_data[i] > _confidence_threshold) {
            indices[count] = i;
            count++;
        }
    }

    int num_output_scores = (_top_k == -1 ? count : std::min<int>(_top_k, count));

    std::partial_sort_copy(indices, indices + count,
                           buffer, buffer + num_output_scores,
                           ConfidenceComparator(conf_data));

    for (int i = 0; i < num_output_scores; ++i) {
        const int idx = buffer[i];

        bool keep = true;
        for (int k = 0; k < detections; ++k) {
            const int kept_idx = indices[k];
            float overlap = JaccardOverlap(bboxes, sizes, idx, kept_idx);
            if (overlap > _nms_threshold) {
                keep = false;
                break;
            }
        }
        if (keep) {
            indices[detections] = idx;
            detections++;
        }
    }
}

