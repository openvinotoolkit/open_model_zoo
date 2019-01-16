// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ext_list.hpp"
#include "ext_base.hpp"

#include <cfloat>
#include <vector>
#include <cmath>
#include <string>
#include <utility>
#include <algorithm>

using namespace InferenceEngine;
using namespace InferenceEngine::Extensions;
using namespace InferenceEngine::Extensions::Cpu;

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
    CNNLayer cnnLayer;

public:
    explicit DetectionOutputPostProcessor(const CNNLayer* layer): cnnLayer(*layer) {
        try {
            if (cnnLayer.insData.size() != 3)
                THROW_IE_EXCEPTION << "Incorrect number of input edges for layer " << cnnLayer.name;
            if (cnnLayer.outData.empty())
                THROW_IE_EXCEPTION << "Incorrect number of output edges for layer " << cnnLayer.name;

            _num_classes = cnnLayer.GetParamAsInt("num_classes");
            _background_label_id = cnnLayer.GetParamAsInt("background_label_id", 0);
            _top_k = cnnLayer.GetParamAsInt("top_k", -1);
            _variance_encoded_in_target = cnnLayer.GetParamsAsBool("variance_encoded_in_target", false);
            _keep_top_k = cnnLayer.GetParamAsInt("keep_top_k", -1);
            _nms_threshold = cnnLayer.GetParamAsFloat("nms_threshold");
            _confidence_threshold = cnnLayer.GetParamAsFloat("confidence_threshold", -FLT_MAX);
            _share_location = cnnLayer.GetParamsAsBool("share_location", true);
            _normalized = cnnLayer.GetParamsAsBool("normalized", true);
            _image_height = cnnLayer.GetParamAsInt("input_height", 1);
            _image_width = cnnLayer.GetParamAsInt("input_width", 1);
            _prior_size = _normalized ? 4 : 5;
            _offset = _normalized ? 0 : 1;
            _num_loc_classes = _share_location ? 1 : _num_classes;

            std::string code_type_str = cnnLayer.GetParamAsString("code_type", "caffe.PriorBoxParameter.CORNER");
            _code_type = (code_type_str == "caffe.PriorBoxParameter.CENTER_SIZE" ? CodeType::CENTER_SIZE
                                                                                 : CodeType::CORNER);

            auto prior_ptr = cnnLayer.insData[idx_priors].lock();
            if (!prior_ptr) {
                THROW_IE_EXCEPTION << "Pointer to the input data of layer " << cnnLayer.name << " is exprired";
            }
            auto prior_dims = prior_ptr->dims;
            int priors_size = prior_dims[0]*prior_dims[1];

            auto loc_ptr = cnnLayer.insData[idx_location].lock();
            if (!loc_ptr) {
                THROW_IE_EXCEPTION << "Pointer to the input data of layer " << cnnLayer.name << " is exprired";
            }
            auto loc_dims = loc_ptr->dims;
            int loc_size = loc_dims[loc_dims.size() - 2]*loc_dims[loc_dims.size() - 1];

            auto conf_ptr = cnnLayer.insData[idx_confidence].lock();
            if (!conf_ptr) {
                THROW_IE_EXCEPTION << "Pointer to the input data of layer " << cnnLayer.name << " is exprired";
            }
            auto conf_dims = conf_ptr->dims;
            uint32_t conf_size = conf_dims[conf_dims.size() - 2]*conf_dims[conf_dims.size() - 1];

            _num_priors = static_cast<int>(priors_size / _prior_size);

            if (_num_priors * _num_loc_classes * 4 != loc_size)
                THROW_IE_EXCEPTION << "Number of priors must match number of location predictions.";

            if (_num_priors * _num_classes != conf_size)
                THROW_IE_EXCEPTION << "Number of priors must match number of confidence predictions.";

            _num = static_cast<int>(conf_size);

            SizeVector bboxes_size{static_cast<size_t>(_num),
                                                    static_cast<size_t>(_num_classes),
                                                    static_cast<size_t>(_num_priors),
                                                    4};
            _decoded_bboxes = make_shared_blob<float>({Precision::FP32, bboxes_size, NCHW});
            _decoded_bboxes->allocate();

            SizeVector buf_size{static_cast<size_t>(_num),
                                                 static_cast<size_t>(_num_classes),
                                                 static_cast<size_t>(_num_priors)};
            _buffer = make_shared_blob<int>({Precision::I32, buf_size, {buf_size, {0, 1, 2}}});
            _buffer->allocate();

            SizeVector indices_size{static_cast<size_t>(_num),
                                                     static_cast<size_t>(_num_classes),
                                                     static_cast<size_t>(_num_priors)};
            _indices = make_shared_blob<int>(
                    {Precision::I32, indices_size, {indices_size, {0, 1, 2}}});
            _indices->allocate();

            SizeVector detections_size{static_cast<size_t>(_num * _num_classes)};
            _detections_count = make_shared_blob<int>({Precision::I32, detections_size, C});
            _detections_count->allocate();

            SizeVector conf_size1 = { conf_size, 1 };
            _reordered_conf = make_shared_blob<float>({Precision::FP32, conf_size1, ANY});
            _reordered_conf->allocate();

            SizeVector decoded_bboxes_size{static_cast<size_t>(_num),
                                                            static_cast<size_t>(_num_priors),
                                                            static_cast<size_t>(_num_classes)};
            _bbox_sizes = make_shared_blob<float>(
                    {Precision::FP32, decoded_bboxes_size, {decoded_bboxes_size, {0, 1, 2}}});
            _bbox_sizes->allocate();

            SizeVector num_priors_actual_size{static_cast<size_t>(_num)};
            _num_priors_actual = make_shared_blob<int>({Precision::I32, num_priors_actual_size, C});
            _num_priors_actual->allocate();
        } catch (const InferenceEngineException& ex) {
            throw std::logic_error(std::string("Can't create detection output: ") + ex.what());
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept {
        float *dst_data = outputs[0]->buffer();

        const float *loc_data    = inputs[idx_location]->buffer();
        const float *conf_data   = inputs[idx_confidence]->buffer();
        const float *prior_data  = inputs[idx_priors]->buffer();

        auto conf_dims = inputs[idx_confidence]->getTensorDesc().getDims();
        uint32_t conf_size = conf_dims[0]*conf_dims[1];

        const int N = 1;  // TODO: Support batch

        float *decoded_bboxes_data = _decoded_bboxes->buffer();
        float *reordered_conf_data = _reordered_conf->buffer();
        float *bbox_sizes_data     = _bbox_sizes->buffer();
        int *detections_data       = _detections_count->buffer();
        int *buffer_data           = _buffer->buffer();
        int *indices_data          = _indices->buffer();
        int *num_priors_actual     = _num_priors_actual->buffer();

        const float *prior_variances = prior_data + _num_priors*_prior_size;
        const float *ppriors = prior_data;

        for (int n = 0; n < N; ++n) {
            if (_share_location) {
                const float *ploc = loc_data + n*4*_num_priors;
                float *pboxes = decoded_bboxes_data + n*4*_num_priors;
                float *psizes = bbox_sizes_data + n*_num_priors;
                decodeBBoxes(ppriors, ploc, prior_variances, pboxes, psizes, num_priors_actual, n);
            } else {
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
                const float *pboxes;
                const float *psizes;
                if (_share_location) {
                    pboxes = decoded_bboxes_data + n*4*_num_priors;
                    psizes = bbox_sizes_data + n*_num_priors;
                } else {
                    pboxes = decoded_bboxes_data + n*4*_num_classes*_num_priors + c*4*_num_priors;
                    psizes = bbox_sizes_data + n*_num_classes*_num_priors + c*_num_priors;
                }

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

                for (int j = 0; j < conf_index_class_map.size(); ++j) {
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

                    dst_data[count * DETECTION_SIZE + 0] = n;
                    dst_data[count * DETECTION_SIZE + 1] = c;
                    dst_data[count * DETECTION_SIZE + 2] = pconf[c*_num_priors + idx];

                    float xmin = _share_location ? pboxes[idx*4 + 0] :
                                 pboxes[c*4*_num_priors + idx*4 + 0];
                    float ymin = _share_location ? pboxes[idx*4 + 1] :
                                 pboxes[c*4*_num_priors + idx*4 + 1];
                    float xmax = _share_location ? pboxes[idx*4 + 2] :
                                 pboxes[c*4*_num_priors + idx*4 + 2];
                    float ymax = _share_location ? pboxes[idx*4 + 3] :
                                 pboxes[c*4*_num_priors + idx*4 + 3];

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
    int _background_label_id = 0;
    int _top_k = 0;
    int _variance_encoded_in_target = 0;
    int _keep_top_k = 0;
    int _code_type = 0;

    bool _share_location = false;

    int _image_width = 0;
    int _image_height = 0;
    int _prior_size = 4;
    bool _normalized = true;
    int _offset = 0;

    float _nms_threshold = 0.0f;
    float _confidence_threshold = 0.0f;

    int _num = 0;
    int _num_loc_classes = 0;
    int _num_priors = 0;

    enum CodeType {
        CORNER = 1,
        CENTER_SIZE = 2,
    };

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
    if (!_normalized) {
        int num = 0;
        for (; num < _num_priors; ++num) {
            float batch_id = prior_data[num * _prior_size + 0];
            if (batch_id == -1.f) {
                num_priors_actual[n] = num;
                break;
            }
        }
    }

    for (int p = 0; p < num_priors_actual[n]; ++p) {
        float new_xmin = 0.0f;
        float new_ymin = 0.0f;
        float new_xmax = 0.0f;
        float new_ymax = 0.0f;

        float prior_xmin = prior_data[p*_prior_size + 0 + _offset];
        float prior_ymin = prior_data[p*_prior_size + 1 + _offset];
        float prior_xmax = prior_data[p*_prior_size + 2 + _offset];
        float prior_ymax = prior_data[p*_prior_size + 3 + _offset];

        float loc_xmin = loc_data[4*p*_num_loc_classes + 0];
        float loc_ymin = loc_data[4*p*_num_loc_classes + 1];
        float loc_xmax = loc_data[4*p*_num_loc_classes + 2];
        float loc_ymax = loc_data[4*p*_num_loc_classes + 3];

        if (!_normalized) {
            prior_xmin /= _image_width;
            prior_ymin /= _image_height;
            prior_xmax /= _image_width;
            prior_ymax /= _image_height;
        }

        if (_code_type == CodeType::CORNER) {
            if (_variance_encoded_in_target) {
                // variance is encoded in target, we simply need to add the offset predictions.
                new_xmin = prior_xmin + loc_xmin;
                new_ymin = prior_ymin + loc_ymin;
                new_xmax = prior_xmax + loc_xmax;
                new_ymax = prior_ymax + loc_ymax;
            } else {
                new_xmin = prior_xmin + variance_data[p*4 + 0] * loc_xmin;
                new_ymin = prior_ymin + variance_data[p*4 + 1] * loc_ymin;
                new_xmax = prior_xmax + variance_data[p*4 + 2] * loc_xmax;
                new_ymax = prior_ymax + variance_data[p*4 + 3] * loc_ymax;
            }
        } else if (_code_type == CodeType::CENTER_SIZE) {
            float prior_width    =  prior_xmax - prior_xmin;
            float prior_height   =  prior_ymax - prior_ymin;
            float prior_center_x = (prior_xmin + prior_xmax) / 2.0f;
            float prior_center_y = (prior_ymin + prior_ymax) / 2.0f;

            float decode_bbox_center_x, decode_bbox_center_y;
            float decode_bbox_width, decode_bbox_height;

            if (_variance_encoded_in_target) {
                // variance is encoded in target, we simply need to restore the offset predictions.
                decode_bbox_center_x = loc_xmin * prior_width  + prior_center_x;
                decode_bbox_center_y = loc_ymin * prior_height + prior_center_y;
                decode_bbox_width  = std::exp(loc_xmax) * prior_width;
                decode_bbox_height = std::exp(loc_ymax) * prior_height;
            } else {
                // variance is encoded in bbox, we need to scale the offset accordingly.
                decode_bbox_center_x = variance_data[p*4 + 0] * loc_xmin * prior_width + prior_center_x;
                decode_bbox_center_y = variance_data[p*4 + 1] * loc_ymin * prior_height + prior_center_y;
                decode_bbox_width    = std::exp(variance_data[p*4 + 2] * loc_xmax) * prior_width;
                decode_bbox_height   = std::exp(variance_data[p*4 + 3] * loc_ymax) * prior_height;
            }

            new_xmin = decode_bbox_center_x - decode_bbox_width  / 2.0f;
            new_ymin = decode_bbox_center_y - decode_bbox_height / 2.0f;
            new_xmax = decode_bbox_center_x + decode_bbox_width  / 2.0f;
            new_ymax = decode_bbox_center_y + decode_bbox_height / 2.0f;
        }

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

