#include "test_precomp.hpp"

#include <opencv2/core/utils/filesystem.hpp>

namespace opencv_test { namespace {

using namespace open_model_zoo;
using namespace open_model_zoo::topologies;

TEST(Topologies, visibility)
{
    auto t = face_detection_retail_fp16();
    std::string bin = t.getModelPath();
    std::string xml = t.getConfigPath();
    ASSERT_EQ(bin.substr(bin.rfind('.')), ".bin");
    ASSERT_EQ(xml.substr(xml.rfind('.')), ".xml");
}

// source: dnn/src/nms.cpp
static inline float rotatedRectIOU(const RotatedRect& a, const RotatedRect& b)
{
    std::vector<Point2f> inter;
    int res = rotatedRectangleIntersection(a, b, inter);
    if (inter.empty() || res == INTERSECT_NONE)
        return 0.0f;
    if (res == INTERSECT_FULL)
        return 1.0f;
    float interArea = contourArea(inter);
    return interArea / (a.size.area() + b.size.area() - interArea);
}

// This test requires downloaded models.
TEST(TextRecognitionPipeline, Accuracy)
{
    auto detection = text_detection();
    auto recognition = text_recognition();
    TextRecognitionPipeline p(detection, recognition);
    p.setPixelLinkThresh(0.5f);
    p.setPixelClassificationThresh(0.5f);

    Mat img = imread(findDataFile("cv/cloning/Mixed_Cloning/source1.png"));

    std::vector<RotatedRect> rects;
    std::vector<std::string> texts;
    std::vector<float> confs;
    p.process(img, rects, texts, confs);

    std::vector<std::string> refTexts = {"c57410", "jie", "howard"};
    std::vector<RotatedRect> refRects = {
      RotatedRect(Point2f(110.39253234863281, 45.5788459777832), Size2f(48.49958419799805, 153.86648559570312), -87.61405944824219),
      RotatedRect(Point2f(93.0, 102.5), Size2f(80.0, 43.0), -0.0),
      RotatedRect(Point2f(111.65045928955078, 152.82647705078125), Size2f(48.7945442199707, 173.10397338867188), -88.87670135498047)
    };

    ASSERT_EQ(texts.size(), rects.size());
    ASSERT_EQ(texts.size(), refTexts.size());

    for (size_t i = 0; i < rects.size(); ++i)
    {
        bool matched = false;
        size_t j = 0;
        for (; j < rects.size(); ++j)
        {
            if (texts[j] == refTexts[i])
            {
                matched = true;
                break;
            }
        }
        ASSERT_TRUE(matched) << refTexts[i];
        ASSERT_GE(rotatedRectIOU(rects[j], refRects[i]), 0.99) << refTexts[i];
    }
}

}}  // namespace opencv_test
