// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <string>
#include <list>
#include <vector>
#include <map>
#include <algorithm>

#include "visualizer.hpp"

// EmotionBarVisualizer
EmotionBarVisualizer::EmotionBarVisualizer(std::vector<std::string> const& emotionNames, cv::Size size, cv::Size padding,
                                     double opacity, double textScale, int textThickness):
                                     emotionNames(emotionNames), size(size), padding(padding),
                                     opacity(opacity), textScale(textScale), textThickness(textThickness),
                                     internalPadding(0) {
    auto itMax = std::max_element(emotionNames.begin(), emotionNames.end(), [] (std::string const& lhs, std::string const& rhs) {
        return lhs.length() < rhs.length();
    });

    textSize = cv::getTextSize(*itMax, cv::FONT_HERSHEY_COMPLEX_SMALL, textScale, textThickness, &textBaseline);
    ystep = (emotionNames.size() < 2) ? 0 : (size.height - 2 * padding.height - textSize.height) / (emotionNames.size() - 1);
}

cv::Size EmotionBarVisualizer::getSize() {
    return size;
}

void EmotionBarVisualizer::draw(cv::Mat& img, std::map<std::string, float> emotions, cv::Point org, cv::Scalar fgcolor, cv::Scalar bgcolor) {
    cv::Mat tmp = img(cv::Rect(org.x, org.y, size.width, size.height));
    cv::addWeighted(tmp, 1.f - opacity, bgcolor, opacity, 0, tmp);

    auto drawEmotion = [&](int n, std::string text, float value) {
        cv::Point torg(org.x + padding.width, org.y + n * ystep + textSize.height + padding.height);

        int textWidth = textSize.width + 10;
        cv::Rect r(torg.x + textWidth, torg.y - textSize.height, size.width - 2 * padding.width - textWidth, textSize.height + textBaseline / 2);

        cv::putText(img, text, torg, cv::FONT_HERSHEY_COMPLEX_SMALL, textScale, fgcolor, textThickness);
        cv::rectangle(img, r, fgcolor, 1);
        r.width = static_cast<int>(r.width * value);
        cv::rectangle(img, r, fgcolor, cv::FILLED);
    };

    for (size_t i = 0; i< emotionNames.size(); i++) {
        drawEmotion(i, emotionNames[i], emotions[emotionNames[i]]);
    }
}

// PhotoFrameVisualizer
PhotoFrameVisualizer::PhotoFrameVisualizer(int bbThickness, int photoFrameThickness, float photoFrameLength):
    bbThickness(bbThickness), photoFrameThickness(photoFrameThickness), photoFrameLength(photoFrameLength) {
}

void PhotoFrameVisualizer::draw(cv::Mat& img, cv::Rect& bb, cv::Scalar color) {
    cv::rectangle(img, bb, color, bbThickness);

    auto drawPhotoFrameCorner = [&](cv::Point p, int dx, int dy) {
        cv::line(img, p, cv::Point(p.x, p.y + dy), color, photoFrameThickness);
        cv::line(img, p, cv::Point(p.x + dx, p.y), color, photoFrameThickness);
    };

    int dx = static_cast<int>(photoFrameLength * bb.width);
    int dy = static_cast<int>(photoFrameLength * bb.height);

    drawPhotoFrameCorner(bb.tl(), dx, dy);
    drawPhotoFrameCorner(cv::Point(bb.x + bb.width - 1, bb.y), -dx, dy);
    drawPhotoFrameCorner(cv::Point(bb.x, bb.y + bb.height - 1), dx, -dy);
    drawPhotoFrameCorner(cv::Point(bb.x + bb.width - 1, bb.y + bb.height - 1), -dx, -dy);
}

// HeadPoseVisualizer
HeadPoseVisualizer::HeadPoseVisualizer(float scale, cv::Scalar xAxisColor, cv::Scalar yAxisColor, cv::Scalar zAxisColor, int axisThickness):
                        xAxisColor(xAxisColor), yAxisColor(yAxisColor), zAxisColor(zAxisColor), axisThickness(axisThickness), scale(scale) {
}

void HeadPoseVisualizer::buildCameraMatrix(cv::Mat& cameraMatrix, int cx, int cy, float focalLength) {
    if (!cameraMatrix.empty()) return;
    cameraMatrix = cv::Mat::zeros(3, 3, CV_32F);
    cameraMatrix.at<float>(0) = focalLength;
    cameraMatrix.at<float>(2) = static_cast<float>(cx);
    cameraMatrix.at<float>(4) = focalLength;
    cameraMatrix.at<float>(5) = static_cast<float>(cy);
    cameraMatrix.at<float>(8) = 1;
}

void HeadPoseVisualizer::draw(cv::Mat& frame, cv::Point3f cpoint, HeadPoseDetection::Results headPose) {
    double yaw   = headPose.angle_y;
    double pitch = headPose.angle_p;
    double roll  = headPose.angle_r;

    pitch *= CV_PI / 180.0;
    yaw   *= CV_PI / 180.0;
    roll  *= CV_PI / 180.0;

    cv::Matx33f Rx(1, 0, 0,
                   0, static_cast<float>(cos(pitch)), static_cast<float>(-sin(pitch)),
                   0, static_cast<float>(sin(pitch)), static_cast<float>(cos(pitch)));

    cv::Matx33f Ry(static_cast<float>(cos(yaw)), 0, static_cast<float>(-sin(yaw)),
                   0, 1, 0,
                   static_cast<float>(sin(yaw)), 0, static_cast<float>(cos(yaw)));

    cv::Matx33f Rz(static_cast<float>(cos(roll)), static_cast<float>(-sin(roll)), 0,
                   static_cast<float>(sin(roll)),  static_cast<float>(cos(roll)), 0,
                   0, 0, 1);


    auto r = cv::Mat(Rz*Ry*Rx);
    cv::Mat cameraMatrix;
    buildCameraMatrix(cameraMatrix, frame.cols / 2, frame.rows / 2, 950.0);

    cv::Mat xAxis(3, 1, CV_32F), yAxis(3, 1, CV_32F), zAxis(3, 1, CV_32F), zAxis1(3, 1, CV_32F);

    xAxis.at<float>(0) = 1 * scale;
    xAxis.at<float>(1) = 0;
    xAxis.at<float>(2) = 0;

    yAxis.at<float>(0) = 0;
    yAxis.at<float>(1) = -1 * scale;
    yAxis.at<float>(2) = 0;

    zAxis.at<float>(0) = 0;
    zAxis.at<float>(1) = 0;
    zAxis.at<float>(2) = -1 * scale;

    zAxis1.at<float>(0) = 0;
    zAxis1.at<float>(1) = 0;
    zAxis1.at<float>(2) = 1 * scale;

    cv::Mat o(3, 1, CV_32F, cv::Scalar(0));
    o.at<float>(2) = cameraMatrix.at<float>(0);

    xAxis = r * xAxis + o;
    yAxis = r * yAxis + o;
    zAxis = r * zAxis + o;
    zAxis1 = r * zAxis1 + o;

    cv::Point p1, p2;

    p2.x = static_cast<int>((xAxis.at<float>(0) / xAxis.at<float>(2) * cameraMatrix.at<float>(0)) + cpoint.x);
    p2.y = static_cast<int>((xAxis.at<float>(1) / xAxis.at<float>(2) * cameraMatrix.at<float>(4)) + cpoint.y);
    cv::line(frame, cv::Point(static_cast<int>(cpoint.x), static_cast<int>(cpoint.y)), p2, xAxisColor, axisThickness);

    p2.x = static_cast<int>((yAxis.at<float>(0) / yAxis.at<float>(2) * cameraMatrix.at<float>(0)) + cpoint.x);
    p2.y = static_cast<int>((yAxis.at<float>(1) / yAxis.at<float>(2) * cameraMatrix.at<float>(4)) + cpoint.y);
    cv::line(frame, cv::Point(static_cast<int>(cpoint.x), static_cast<int>(cpoint.y)), p2, yAxisColor, axisThickness);

    p1.x = static_cast<int>((zAxis1.at<float>(0) / zAxis1.at<float>(2) * cameraMatrix.at<float>(0)) + cpoint.x);
    p1.y = static_cast<int>((zAxis1.at<float>(1) / zAxis1.at<float>(2) * cameraMatrix.at<float>(4)) + cpoint.y);

    p2.x = static_cast<int>((zAxis.at<float>(0) / zAxis.at<float>(2) * cameraMatrix.at<float>(0)) + cpoint.x);
    p2.y = static_cast<int>((zAxis.at<float>(1) / zAxis.at<float>(2) * cameraMatrix.at<float>(4)) + cpoint.y);
    cv::line(frame, p1, p2, zAxisColor, axisThickness);
    cv::circle(frame, p2, 3, zAxisColor, axisThickness);
}

// Visualizer
Visualizer::Visualizer(cv::Size const& imgSize, int leftPadding, int rightPadding, int topPadding, int bottomPadding):
        emotionVisualizer(nullptr), photoFrameVisualizer(std::make_shared<PhotoFrameVisualizer>()),
        headPoseVisualizer(std::make_shared<HeadPoseVisualizer>()),
        nxcells(0), nycells(0), xstep(0), ystep(0), imgSize(imgSize), leftPadding(leftPadding),
        rightPadding(rightPadding), topPadding(topPadding), bottomPadding(bottomPadding), frameCounter(0) {}

void Visualizer::enableEmotionBar(std::vector<std::string> const& emotionNames) {
    emotionVisualizer = std::make_shared<EmotionBarVisualizer>(emotionNames);
    emotionBarSize = emotionVisualizer->getSize();

    cv::Size imgSizePadded;
    imgSizePadded.width = imgSize.width - leftPadding - rightPadding;
    imgSizePadded.height = imgSize.height - topPadding - bottomPadding;

    nxcells = (imgSizePadded.width - 1) / emotionBarSize.width;
    nycells = (imgSizePadded.height - 1) / emotionBarSize.height;
    if (0 < nxcells && 0 < nycells) {
        drawMap.create(nycells, nxcells, CV_8UC1);

        xstep = imgSizePadded.width / nxcells;
        ystep = imgSizePadded.height / nycells;
    } else {
        emotionVisualizer.reset();
        std::cerr << "Disabling emotion bar due to small frame resolution to draw on\n";
    }
}

void Visualizer::drawFace(cv::Mat& img, Face::Ptr f, bool drawEmotionBar) {
    auto genderColor = (f->isAgeGenderEnabled()) ?
                       ((f->isMale()) ? cv::Scalar(255, 0, 0) :
                                        cv::Scalar(147, 20, 255)) :
                                        cv::Scalar(100, 100, 100);

    std::ostringstream out;
    if (f->isAgeGenderEnabled()) {
        out << (f->isMale() ? "Male" : "Female");
        out << "," << f->getAge();
    }

    if (f->isEmotionsEnabled()) {
        auto emotion = f->getMainEmotion();
        out << "," << emotion.first;
    }

    cv::putText(img,
                out.str(),
                cv::Point2f(static_cast<float>(f->_location.x), static_cast<float>(f->_location.y - 20)),
                cv::FONT_HERSHEY_COMPLEX_SMALL,
                1.5,
                genderColor, 2);

    if (f->isHeadPoseEnabled()) {
        cv::Point3f center(static_cast<float>(f->_location.x + f->_location.width / 2),
                           static_cast<float>(f->_location.y + f->_location.height / 2),
                           0.0f);
        headPoseVisualizer->draw(img, center, f->getHeadPose());
    }

    if (f->isLandmarksEnabled()) {
        auto& normed_landmarks = f->getLandmarks();
        size_t n_lm = normed_landmarks.size();
        for (size_t i_lm = 0UL; i_lm < n_lm / 2; ++i_lm) {
            float normed_x = normed_landmarks[2 * i_lm];
            float normed_y = normed_landmarks[2 * i_lm + 1];

            int x_lm = f->_location.x + static_cast<int>(f->_location.width * normed_x);
            int y_lm = f->_location.y + static_cast<int>(f->_location.height * normed_y);
            cv::circle(img, cv::Point(x_lm, y_lm), 1 + static_cast<int>(0.012 * f->_location.width), cv::Scalar(0, 255, 255), -1);
        }
    }

    photoFrameVisualizer->draw(img, f->_location, genderColor);

    if (drawEmotionBar) {
        DrawParams& dp = drawParams[f->getId()];
        cv::Point org(dp.cell.x * xstep + leftPadding, imgSize.height - dp.cell.y * ystep - emotionBarSize.height - bottomPadding);

        emotionVisualizer->draw(img, f->getEmotions(), org, cv::Scalar(255, 255, 255), genderColor);

        auto getCorner = [](cv::Rect r, AnchorType anchor) -> cv::Point {
            cv::Point p;
            if (anchor == AnchorType::TL) {
                p = r.tl();
            } else if (anchor == AnchorType::TR) {
                p.x = r.x + r.width - 1;
                p.y = r.y;
            } else if (anchor == AnchorType::BL) {
                p.x = r.x;
                p.y = r.y + r.height - 1;
            } else {
                p.x = r.x + r.width - 1;
                p.y = r.y + r.height - 1;
            }

            return p;
        };

        cv::Point p0 = getCorner(cv::Rect(org, emotionBarSize), dp.barAnchor);
        cv::Point p1 = getCorner(f->_location, dp.rectAnchor);
        cv::line(img, p0, p1, genderColor);
    }
}

cv::Point Visualizer::findCellForEmotionBar() {
    cv::Point p;
    int yEnd = std::max(nycells / 2, 1);
    for (p.y = 0; p.y < yEnd; p.y++) {
        for (p.x = 0; p.x < nxcells; p.x++) {
            if (drawMap.at<uchar>(p.y, p.x) == 0) {
                return p;
            }
        }
    }

    for (p.x = 0, p.y = yEnd; p.y < nycells; p.y++) {
        if (drawMap.at<uchar>(p.y, p.x) == 0) {
            return p;
        }
    }

    for (p.x = nxcells - 1, p.y = yEnd; p.y < nycells; p.y++) {
        if (drawMap.at<uchar>(p.y, p.x) == 0) {
            return p;
        }
    }

    return cv::Point(-1, -1);
}

void Visualizer::draw(cv::Mat img, std::list<Face::Ptr> faces) {
    drawMap.setTo(0);
    frameCounter++;

    std::vector<Face::Ptr> newFaces;
    for (auto&& face : faces) {
        if (emotionVisualizer) {
            if (drawParams.find(face->getId()) == drawParams.end()) {
                newFaces.push_back(face);
                continue;
            }

            drawFace(img, face, true);

            drawParams[face->getId()].frameIdx = frameCounter;

            cv::Point& cell = drawParams[face->getId()].cell;
            drawMap.at<uchar>(cell.y, cell.x) = 1;
        } else {
            drawFace(img, face, false);
        }
    }

    if (!newFaces.empty()) {
        auto it = drawParams.begin();
        auto endIt = drawParams.end();
        for (; it != endIt; ) {
            if (it->second.frameIdx != frameCounter) {
                it = drawParams.erase(it);
            } else {
                ++it;
            }
        }

        for (auto&& face : newFaces) {
            DrawParams dp;
            dp.cell = findCellForEmotionBar();

            if ((dp.cell.x < 0) || (dp.cell.y < 0)) {
                drawFace(img, face, false);
            } else {
                int nycells2 = (nycells + 1) / 2;
                int nxcells2 = (nxcells + 1) / 2;
                if ((dp.cell.x < nxcells2) && (dp.cell.y < nycells2)) {
                    dp.barAnchor = AnchorType::TR;
                    dp.rectAnchor = AnchorType::BL;
                } else if ((dp.cell.x >= nxcells2) && (dp.cell.y < nycells2)) {
                    dp.barAnchor = AnchorType::TL;
                    dp.rectAnchor = AnchorType::BR;
                } else if ((dp.cell.x < nxcells2) && (dp.cell.y >= nycells2)) {
                    dp.barAnchor = AnchorType::BR;
                    dp.rectAnchor = AnchorType::TL;
                } else {
                    dp.barAnchor = AnchorType::BL;
                    dp.rectAnchor = AnchorType::TR;
                }
                dp.frameIdx = frameCounter;
                drawParams[face->getId()] = dp;

                drawFace(img, face, true);

                cv::Point& cell = drawParams[face->getId()].cell;
                drawMap.at<uchar>(cell.y, cell.x) = 1;
            }
        }
    }
}
