// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom_kernels.hpp"
#include "shared_functions.hpp"
#include "custom_nets.hpp"

#include <opencv2/imgproc.hpp>

void softmax(std::vector<float>& rdata) {
    const size_t lastDim = 2;
    for (size_t i = 0 ; i < rdata.size(); i+=lastDim) {
        float m = std::max(rdata[i], rdata[i+1]);
        rdata[i] = std::exp(rdata[i] - m);
        rdata[i + 1] = std::exp(rdata[i + 1] - m);
        float s = rdata[i] + rdata[i + 1];
        rdata[i] /= s;
        rdata[i + 1] /= s;
    }
}

std::vector<float> transpose4d(const std::vector<float>& data, const std::vector<size_t>& shape,
                               const std::vector<size_t>& axes) {
    if (shape.size() != axes.size()) {
        throw std::runtime_error("Shape and axes must have the same dimension.");
    }

    for (size_t a : axes) {
        if (a >= shape.size()) {
            throw std::runtime_error("Axis must be less than dimension of shape.");
        }
    }
    size_t totalSize = shape[0]*shape[1]*shape[2]*shape[3];
    std::vector<size_t> steps {
        shape[axes[1]]*shape[axes[2]]*shape[axes[3]],
        shape[axes[2]]*shape[axes[3]],
        shape[axes[3]],
        1
    };
    size_t sourceDataIdx = 0;
    std::vector<float> newData(totalSize, 0);
    std::vector<size_t> ids(shape.size());
    for (ids[0] = 0; ids[0] < shape[0]; ids[0]++) {
        for (ids[1] = 0; ids[1] < shape[1]; ids[1]++) {
            for (ids[2] = 0; ids[2] < shape[2]; ids[2]++) {
                for (ids[3]= 0; ids[3] < shape[3]; ids[3]++) {
                    size_t newDataIdx = ids[axes[0]]*steps[0] + ids[axes[1]]*steps[1] +
                        ids[axes[2]]*steps[2] + ids[axes[3]]*steps[3];
                    newData[newDataIdx] = data[sourceDataIdx++];
                }
            }
        }
    }
    return newData;
}

std::vector<std::size_t> dimsToShape(const cv::MatSize& sz) {
    const int nDims = sz.dims();
    std::vector<std::size_t> result(nDims, 0);
    // cv::MatSize is not iterable...
    for (int i = 0; i < nDims; i++) {
        result[i] = static_cast<std::size_t>(sz[i]);
    }
    return result;
}

std::vector<float> sliceAndGetSecondChannel(const std::vector<float>& data) {
    std::vector<float> newData(data.size() / 2, 0);
    for (size_t i = 0; i < data.size() / 2; i++) {
        newData[i] = data[2 * i + 1];
    }
    return newData;
}

std::vector<cv::RotatedRect> maskToBoxes(const cv::Mat& mask, const float minArea,
                                         const float minHeight, const cv::Size& imgSize) {
    std::vector<cv::RotatedRect> bboxes;
    double minVal = 0.;
    double maxVal = 0.;
    cv::minMaxLoc(mask, &minVal, &maxVal);
    int maxBboxIdx = static_cast<int>(maxVal);
    cv::Mat resizedMask;
    cv::resize(mask, resizedMask, imgSize, 0, 0, cv::INTER_NEAREST);

    for (int i = 1; i <= maxBboxIdx; i++) {
        cv::Mat bboxMask = resizedMask == i;
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(bboxMask, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
        if (contours.empty())
            continue;
        cv::RotatedRect r = cv::minAreaRect(contours[0]);
        if (std::min(r.size.width, r.size.height) < minHeight)
            continue;
        if (r.size.area() < minArea)
            continue;
        bboxes.emplace_back(r);
    }
    return bboxes;
}

std::vector<cv::RotatedRect> coordToBoxes(const float* coords, size_t coordsSize,
                                          float minArea, float minHeight,
                                          const cv::Size& inputShape, const cv::Size& imgSize) {
    std::vector<cv::RotatedRect> bboxes;
    int numBoxes = coordsSize / 5;
    float xScale = imgSize.width / float(inputShape.width);
    float yScale = imgSize.height / float(inputShape.height);
    for (int i = 0; i < numBoxes; i++) {
        const float *prediction = &coords[i * 5];
        float confidence = prediction[4];
        if (confidence < std::numeric_limits<float>::epsilon()) break;
        // predictions are sorted the way that all insignificant boxes are
        // grouped together
        cv::Point2f center = cv::Point2f((prediction[0] + prediction[2]) / 2 * xScale,
                                         (prediction[1] + prediction[3]) / 2 * yScale);
        cv::Size2f size = cv::Size2f((prediction[2] - prediction[0]) * xScale,
                                     (prediction[3] - prediction[1]) * yScale);
        cv::RotatedRect rect = cv::RotatedRect(center, size, 0);

        if (rect.size.area() < minArea) continue;
        bboxes.push_back(rect);
    }
    return bboxes;
}

int findRoot(const int point, std::unordered_map<int, int>& groupMask) {
    int root = point;
    bool updateParent = false;
    while (groupMask.at(root) != -1) {
        root = groupMask.at(root);
        updateParent = true;
    }
    if (updateParent) {
        groupMask[point] = root;
    }
    return root;
}

void join(const int p1, const int p2, std::unordered_map<int, int>& groupMask) {
    const int root1 = findRoot(p1, groupMask);
    const int root2 = findRoot(p2, groupMask);
    if (root1 != root2) {
        groupMask[root1] = root2;
    }
}

cv::Mat getAll(const std::vector<cv::Point>& points, const int w, const int h,
               std::unordered_map<int, int>& groupMask) {
    std::unordered_map<int, int> rootMap;
    cv::Mat mask(h, w, CV_32S, cv::Scalar(0));
    for (const auto &point : points) {
        int pointRoot = findRoot(point.x + point.y * w, groupMask);
        if (rootMap.find(pointRoot) == rootMap.end()) {
            rootMap.emplace(pointRoot, static_cast<int>(rootMap.size() + 1));
        }
        mask.at<int>(point.x + point.y * w) = rootMap[pointRoot];
    }
    return mask;
}

cv::Mat decodeImageByJoin(const std::vector<float>& segmData,
                          const std::vector<int>&   segmDataShape,
                          const std::vector<float>& linkData,
                          const std::vector<int>&   linkDataShape,
                          float segmConfThreshold,
                          float linkConfThreshold) {
    const int h = segmDataShape[1];
    const int w = segmDataShape[2];
    std::vector<uchar> pixelMask(h * w, 0);
    std::unordered_map<int, int> groupMask;
    std::vector<cv::Point> points;
    for (size_t i = 0; i < pixelMask.size(); i++) {
        pixelMask[i] = segmData[i] >= segmConfThreshold;
        if (pixelMask[i]) {
            points.emplace_back(i % w, i / w);
            groupMask[i] = -1;
        }
    }
    std::vector<uchar> linkMask(linkData.size(), 0);
    for (size_t i = 0; i < linkMask.size(); i++) {
        linkMask[i] = linkData[i] >= linkConfThreshold;
    }
    size_t neighbours = size_t(linkDataShape[3]);
    for (const auto &point : points) {
        size_t neighbour = 0;
        for (int ny = point.y - 1; ny <= point.y + 1; ny++) {
            for (int nx = point.x - 1; nx <= point.x + 1; nx++) {
                if (nx == point.x && ny == point.y)
                    continue;
                if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                    uchar pixelValue = pixelMask[size_t(ny) * size_t(w) + size_t(nx)];
                    uchar linkValue = linkMask[(size_t(point.y) * size_t(w) + size_t(point.x))
                                                    *neighbours + neighbour];
                    if (pixelValue && linkValue) {
                        join(point.x + point.y * w, nx + ny * w, groupMask);
                    }
                }
                neighbour++;
            }
        }
    }
    return getAll(points, w, h, groupMask);
}

void processData(const float* ptr, const size_t dataTotal, const cv::MatSize& dataShape,
                 std::vector<float>& out, std::vector<int>& newShape) {
    std::vector<float> inData(ptr, ptr + dataTotal);
    inData = transpose4d(inData, dimsToShape(dataShape), {0, 2, 3, 1});
    softmax(inData);
    out = sliceAndGetSecondChannel(inData);
    newShape = { dataShape[0], dataShape[2], dataShape[3], dataShape[1] / 2 };
}

GAPI_OCV_KERNEL(OCVDetectionPostProcess, custom::DetectionPostProcess) {
    static void run(const cv::Mat& det1,
                    const cv::Mat& det2,
                    const cv::Size& imgSize,
                    const cv::Size& inputSize,
                    const float linkThr,
                    const float segmThr,
                    const size_t maxRectsNum,
                          std::vector<cv::RotatedRect> &out) {
        const float kMinArea = 300.0f;
        const float kMinHeight = 10.0f;
        const size_t tdLinkLayerChannels = 16;
        if (det1.size[1] == tdLinkLayerChannels) {
            std::vector<float> linkData{}, segmData{};
            std::vector<int> linkNewShape{}, segmNewShape{};
            processData(det1.ptr<float>(), det1.total(), det1.size, linkData, linkNewShape);
            processData(det2.ptr<float>(), det2.total(), det2.size, segmData, segmNewShape);
            auto mask = decodeImageByJoin(segmData, segmNewShape, linkData, linkNewShape,
                                          segmThr, linkThr);
            out = maskToBoxes(mask, kMinArea, kMinHeight, imgSize);
        } else {
            out = coordToBoxes(det1.ptr<float>(), det1.total(), kMinArea, kMinHeight,
                               inputSize, imgSize);
        }
        if (maxRectsNum >= 0 && out.size() > maxRectsNum) {
            std::sort(out.begin(), out.end(),
                      [](const cv::RotatedRect& a, const cv::RotatedRect& b) {
                          return a.size.area() > b.size.area();
                      });
            out.resize(maxRectsNum);
        }
    }
};

cv::Rect centralCropRect(const int imgWidth, const int imgHeight) {
    int w = static_cast<int>(imgWidth * 0.05);
    int h = static_cast<int>(w * 0.5);
    return { static_cast<int>(imgWidth  * 0.5 - w * 0.5),
             static_cast<int>(imgHeight * 0.5 - h * 0.5),
             w, h };
}

std::vector<cv::Point2f> centralCropPoints(const cv::Rect& r) {
    return { r.tl(), r.tl() + cv::Point{ r.width,  0 },
             r.br(), r.tl() + cv::Point{ 0, r.height } };
}
std::vector<cv::Point2f> centralCropPoints(const int imgWidth, const int imgHeight) {
    return centralCropPoints(centralCropRect(imgWidth, imgHeight));
}

GAPI_OCV_KERNEL(OCVPointsFromRRects, custom::PointsFromRRects) {
    static void run(const std::vector<cv::RotatedRect>& rrs,
                    const cv::Size& imgSize,
                    const bool centralCrop,
                          std::vector<std::vector<cv::Point2f>>& pts) {
        pts.clear();
        pts.reserve(rrs.size());
        const auto imgPts = centralCrop
            ? centralCropPoints(imgSize.width, imgSize.height)
            : std::vector<cv::Point2f>{ { 0.0f,                 0.0f                  },
                                        { imgSize.width - 1.0f, 0.0f                  },
                                        { imgSize.width - 1.0f, imgSize.height - 1.0f },
                                        { 0.0f,                 imgSize.height - 1.0f } };
        std::vector<cv::Point2f> points(4);
        std::transform(rrs.begin(), rrs.end(), std::back_inserter(pts),
                       [&points, &imgPts](const cv::RotatedRect& rr) {
                           if (rr.size != cv::Size2f(0, 0)) {
                               rr.points(points.data());
                               return std::vector<cv::Point2f>(points);
                           } else {
                               return std::vector<cv::Point2f>(imgPts);
                           }
                        });
    }
};

void cropImage(const cv::Mat& image, const std::vector<cv::Point2f>& points,
               const cv::Size& targetSize, cv::Mat& crop) {
    const size_t topLeftPointIdx = custom::getTopLeftPointIdx(points);
    cv::Point2f point0 = points[topLeftPointIdx];
    cv::Point2f point1 = points[(topLeftPointIdx + 1) % 4];
    cv::Point2f point2 = points[(topLeftPointIdx + 2) % 4];
    std::vector<cv::Point2f> from { point0, point1, point2 };

    const float targetWidth  = static_cast<float>(targetSize.width  - 1);
    const float targetHeight = static_cast<float>(targetSize.height - 1);
    std::vector<cv::Point2f> to { cv::Point2f(0.0f,        0.0f),
                                  cv::Point2f(targetWidth, 0.0f),
                                  cv::Point2f(targetWidth, targetHeight) };

    cv::warpAffine(image, crop, cv::getAffineTransform(from, to), targetSize);
}

GAPI_OCV_KERNEL(OCVCropLabels, custom::CropLabels) {
    static void run(const cv::Mat& image,
                    const std::vector<cv::RotatedRect>& rrs,
                    const std::vector<size_t>& outDims,
                    const bool centralCrop,
                          std::vector<cv::Mat>& out,
                          std::vector<std::vector<cv::Point2f>>& pts) {
        cv::Size outSize(outDims[3], outDims[2]);
        std::vector<int> blobShape(4);
        std::copy(outDims.begin(), outDims.end(), blobShape.begin());
        const std::vector<cv::Point2f> imgPts { { 0.0f,              0.0f              },
                                                { image.cols - 1.0f, 0.0f              },
                                                { image.cols - 1.0f, image.rows - 1.0f },
                                                { 0.0f,              image.rows - 1.0f } };
        out.clear();
        out.reserve(rrs.size());
        pts.clear();
        pts.reserve(rrs.size());
        cv::Mat crop(outSize,   CV_8UC3,  cv::Scalar(0));
        cv::Mat blob(blobShape, CV_32FC1, cv::Scalar(0));
        std::vector<cv::Point2f> points(4);
        for (auto& rr : rrs) {
            if (rr.size != cv::Size2f(0, 0)) {
                rr.points(points.data());
                cropImage(image, points, outSize, crop);
            } else {
                if (centralCrop) {
                    auto r = centralCropRect(image.cols, image.rows);
                    points = centralCropPoints(r);
                    cv::resize(image(r), crop, outSize);
                } else {
                    points = imgPts;
                    cv::resize(image, crop, outSize);
                }
            }
            crop.reshape(1, blobShape).convertTo(blob, CV_32F);
            out.emplace_back(blob.clone());
            pts.emplace_back(points);
        }
    }
};

GAPI_OCV_KERNEL(OCVCompositeTRDecode, custom::CompositeTRDecode) {
    static void run(const std::vector<cv::Mat>&  hiddens_,
                    const std::vector<cv::Mat>&  features,
                    const cv::gapi::GNetPackage& net,
                    const size_t                 numClasses,
                    const size_t                 endToken,
                          std::vector<cv::Mat>&  res) {
        constexpr int maxDecodedSymbols = 20;
        GAPI_DbgAssert(hiddens_.size() == features.size());
        const size_t numDetectedLabels = features.size();

        std::vector<cv::Mat> states(numDetectedLabels, cv::Mat({1}, CV_32FC1));
        std::for_each(states.begin(), states.end(),
                      [](cv::Mat& m){ m.dims = 1; *(m.begin<float>()) = 0; });
        std::vector<cv::Mat> hiddens(hiddens_);

        res.reserve(numDetectedLabels);
        for (auto i = 0U; i < numDetectedLabels; i++) {
            res.emplace_back(cv::Mat({ maxDecodedSymbols, 1, static_cast<int>(numClasses) },
                                     CV_32FC1));
        }

        cv::GMat inDec, inHidden, feature, outHidden, outDec;
        std::tie(outHidden, outDec) =
            cv::gapi::infer<nets::TextRecognitionDecoding>(inDec, inHidden, feature);
        static cv::GComputation graph(cv::GIn(inDec, inHidden, feature),
                                      cv::GOut(outHidden, outDec));
        static auto pipeline = graph.compile(
            cv::descr_of(cv::gin(states[0], hiddens[0], features[0])), cv::compile_args(net));

        cv::Mat hidden, out;
        for (size_t nLabel = 0; nLabel < numDetectedLabels; nLabel++) {
            for (int nSymbol = 0; nSymbol < maxDecodedSymbols; nSymbol++) {
                pipeline(cv::gin(states[nLabel], hiddens[nLabel], features[nLabel]),
                         cv::gout(hidden, out));

                auto maxElem = std::max_element(out.begin<float>(),
                                                out.begin<float>() + numClasses);
                auto argmax = static_cast<size_t>(std::distance(out.begin<float>(), maxElem));
                for (size_t i = 0; i < numClasses; i++) {
                    res[nLabel].begin<float>()[nSymbol * numClasses + i] = out.begin<float>()[i];
                }
                if (endToken == argmax) {
                    break;
                }
                *(states[nLabel].begin<float>()) = float(argmax);
                hiddens[nLabel] = hidden;
            }
        }
    }
};

cv::gapi::GKernelPackage custom::kernels() {
    return cv::gapi::kernels<OCVDetectionPostProcess,
                             OCVPointsFromRRects,
                             OCVCropLabels,
                             OCVCompositeTRDecode>();
}
