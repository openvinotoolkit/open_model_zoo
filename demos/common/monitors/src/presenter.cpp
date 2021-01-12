// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cctype>
#include <chrono>
#include <iomanip>
#include <numeric>

#include "monitors/presenter.h"

namespace {
const std::map<int, MonitorType> keyToMonitorType{
    {'C', MonitorType::CpuAverage},
    {'D', MonitorType::DistributionCpu},
    {'M', MonitorType::Memory}};

std::set<MonitorType> strKeysToMonitorSet(const std::string& keys) {
    std::set<MonitorType> enabledMonitors;
    for (unsigned char key: keys) {
        auto iter = keyToMonitorType.find(std::toupper(key));
        if (keyToMonitorType.end() == iter) {
            throw std::runtime_error("Unknown monitor type");
        } else {
            enabledMonitors.insert(iter->second);
        }
    }
    return enabledMonitors;
}
}

Presenter::Presenter(std::set<MonitorType> enabledMonitors,
        int yPos,
        cv::Size graphSize,
        std::size_t historySize) :
            yPos{yPos},
            graphSize{graphSize},
            graphPadding{std::max(1, static_cast<int>(graphSize.width * 0.05))},
            historySize{historySize},
            distributionCpuEnabled{false},
            strStream{std::ios_base::app} {
    for (MonitorType monitor : enabledMonitors) {
        addRemoveMonitor(monitor);
    }
}

Presenter::Presenter(const std::string& keys, int yPos, cv::Size graphSize, std::size_t historySize) :
    Presenter{strKeysToMonitorSet(keys), yPos, graphSize, historySize} {}

void Presenter::addRemoveMonitor(MonitorType monitor) {
    unsigned updatedHistorySize = 1;
    if (historySize > 1) {
        int sampleStep = std::max(1, static_cast<int>(graphSize.width / (historySize - 1)));
        // +1 to plot graphSize.width/sampleStep segments
        // add round up to and an extra element if don't reach graph edge
        updatedHistorySize = (graphSize.width + sampleStep - 1) / sampleStep + 1;
    }
    switch(monitor) {
        case MonitorType::CpuAverage: {
            if (cpuMonitor.getHistorySize() > 1 && distributionCpuEnabled) {
                cpuMonitor.setHistorySize(1);
            } else if (cpuMonitor.getHistorySize() > 1 && !distributionCpuEnabled) {
                cpuMonitor.setHistorySize(0);
            } else { // cpuMonitor.getHistorySize() <= 1
                cpuMonitor.setHistorySize(updatedHistorySize);
            }
            break;
        }
        case MonitorType::DistributionCpu: {
            if (distributionCpuEnabled) {
                distributionCpuEnabled = false;
                if (1 == cpuMonitor.getHistorySize()) { // cpuMonitor was used only for DistributionCpu => disable it
                    cpuMonitor.setHistorySize(0);
                }
            } else {
                distributionCpuEnabled = true;
                cpuMonitor.setHistorySize(std::max(std::size_t{1}, cpuMonitor.getHistorySize()));
            }
            break;
        }
        case MonitorType::Memory: {
            if (memoryMonitor.getHistorySize() > 1) {
                memoryMonitor.setHistorySize(0);
            } else {
                memoryMonitor.setHistorySize(updatedHistorySize);
            }
            break;
        }
    }
}

void Presenter::handleKey(int key) {
    key = std::toupper(key);
    if ('H' == key) {
        if (0 == cpuMonitor.getHistorySize() && memoryMonitor.getHistorySize() <= 1) {
            addRemoveMonitor(MonitorType::CpuAverage);
            addRemoveMonitor(MonitorType::DistributionCpu);
            addRemoveMonitor(MonitorType::Memory);
        } else {
            cpuMonitor.setHistorySize(0);
            distributionCpuEnabled = false;
            memoryMonitor.setHistorySize(0);
        }
    } else {
        auto iter = keyToMonitorType.find(key);
        if (keyToMonitorType.end() != iter) {
            addRemoveMonitor(iter->second);
        }
    }
}

void Presenter::drawGraphs(cv::Mat& frame) {
    const std::chrono::steady_clock::time_point curTimeStamp = std::chrono::steady_clock::now();
    if (curTimeStamp - prevTimeStamp >= std::chrono::milliseconds{1000}) {
        prevTimeStamp = curTimeStamp;
        if (0 != cpuMonitor.getHistorySize()) {
            cpuMonitor.collectData();
        }
        if (memoryMonitor.getHistorySize() > 1) {
            memoryMonitor.collectData();
        }
    }

    int numberOfEnabledMonitors = (cpuMonitor.getHistorySize() > 1) + distributionCpuEnabled
        + (memoryMonitor.getHistorySize() > 1);
    int panelWidth = graphSize.width * numberOfEnabledMonitors
        + std::max(0, numberOfEnabledMonitors - 1) * graphPadding;
    while (panelWidth > frame.cols) {
        panelWidth = std::max(0, panelWidth - graphSize.width - graphPadding);
        --numberOfEnabledMonitors; // can't draw all monitors
    }
    int graphPos = std::max(0, (frame.cols - 1 - panelWidth) / 2);
    int textGraphSplittingLine = graphSize.height / 5;
    int graphRectHeight = graphSize.height - textGraphSplittingLine;
    int sampleStep = 1;
    unsigned possibleHistorySize = 1;
    if (historySize > 1) {
        sampleStep = std::max(1, static_cast<int>(graphSize.width / (historySize - 1)));
        possibleHistorySize = (graphSize.width + sampleStep - 1) / sampleStep + 1;
    }

    if (cpuMonitor.getHistorySize() > 1 && possibleHistorySize > 1 && --numberOfEnabledMonitors >= 0) {
        std::deque<std::vector<double>> lastHistory = cpuMonitor.getLastHistory();
        cv::Rect intersection = cv::Rect{cv::Point(graphPos, yPos), graphSize} & cv::Rect{0, 0, frame.cols, frame.rows};
        if (!intersection.area()) {
            return;
        }
        cv::Mat graph = frame(intersection);
        graph = graph / 2 + cv::Scalar{127, 127, 127};

        int lineXPos = graph.cols - 1;
        std::vector<cv::Point> averageLoad(lastHistory.size());

        for (int i = lastHistory.size() - 1; i >= 0; --i) {
            double mean = std::accumulate(lastHistory[i].begin(), lastHistory[i].end(), 0.0) / lastHistory[i].size();
            averageLoad[i] = {lineXPos, graphSize.height - static_cast<int>(mean * graphRectHeight)};
            lineXPos -= sampleStep;
        }

        cv::polylines(graph, averageLoad, false, {255, 0, 0}, 2);
        cv::rectangle(frame, cv::Rect{
                cv::Point{graphPos, yPos + textGraphSplittingLine},
                cv::Size{graphSize.width, graphSize.height - textGraphSplittingLine}
            }, {0, 0, 0});
        strStream.str("CPU");
        if (!lastHistory.empty()) {
            strStream << ": " << std::fixed << std::setprecision(1)
                << std::accumulate(lastHistory.back().begin(), lastHistory.back().end(), 0.0)
                    / lastHistory.back().size() * 100 << '%';
        }
        int baseline;
        int textWidth = cv::getTextSize(strStream.str(),
            cv::FONT_HERSHEY_SIMPLEX,
            textGraphSplittingLine * 0.04,
            1,
            &baseline).width;
        cv::putText(graph,
            strStream.str(),
            cv::Point{(graphSize.width - textWidth) / 2, textGraphSplittingLine - 1},
            cv::FONT_HERSHEY_SIMPLEX,
            textGraphSplittingLine * 0.04,
            {70, 0, 0},
            1);
        graphPos += graphSize.width + graphPadding;
    }

    if (distributionCpuEnabled && --numberOfEnabledMonitors >= 0) {
        std::deque<std::vector<double>> lastHistory = cpuMonitor.getLastHistory();
        cv::Rect intersection = cv::Rect{cv::Point(graphPos, yPos), graphSize} & cv::Rect{0, 0, frame.cols, frame.rows};
        if (!intersection.area()) {
            return;
        }
        cv::Mat graph = frame(intersection);
        graph = graph / 2 + cv::Scalar{127, 127, 127};

        if (!lastHistory.empty()) {
            int rectXPos = 0;
            int step = (graph.cols + lastHistory.back().size() - 1) / lastHistory.back().size(); // round up
            double sum = 0;
            for (double coreLoad : lastHistory.back()) {
                sum += coreLoad;
                int height = static_cast<int>(graphRectHeight * coreLoad);
                cv::Rect pillar{cv::Point{rectXPos, graph.rows - height}, cv::Size{step, height}};
                cv::rectangle(graph, pillar, {255, 0, 0}, cv::FILLED);
                cv::rectangle(graph, pillar, {0, 0, 0});
                rectXPos += step;
            }
            sum /= lastHistory.back().size();
            int yLine = graph.rows - static_cast<int>(graphRectHeight * sum);
            cv::line(graph, cv::Point{0, yLine}, cv::Point{graph.cols, yLine}, {0, 255, 0}, 2);
        }
        cv::Rect border{cv::Point{graphPos, yPos + textGraphSplittingLine},
            cv::Size{graphSize.width, graphSize.height - textGraphSplittingLine}};
        cv::rectangle(frame, border, {0, 0, 0});
        strStream.str("Core load");
        if (!lastHistory.empty()) {
            strStream << ": " << std::fixed << std::setprecision(1)
                << std::accumulate(lastHistory.back().begin(), lastHistory.back().end(), 0.0)
                    / lastHistory.back().size() * 100 << '%';
        }
        int baseline;
        int textWidth = cv::getTextSize(strStream.str(),
            cv::FONT_HERSHEY_SIMPLEX,
            textGraphSplittingLine * 0.04,
            1,
            &baseline).width;
        cv::putText(graph,
            strStream.str(),
            cv::Point{(graphSize.width - textWidth) / 2, textGraphSplittingLine - 1},
            cv::FONT_HERSHEY_SIMPLEX,
            textGraphSplittingLine * 0.04,
            {0, 70, 0});
        graphPos += graphSize.width + graphPadding;
    }

    if (memoryMonitor.getHistorySize() > 1 && possibleHistorySize > 1 && --numberOfEnabledMonitors >= 0) {
        std::deque<std::pair<double, double>> lastHistory = memoryMonitor.getLastHistory();
        cv::Rect intersection = cv::Rect{cv::Point(graphPos, yPos), graphSize} & cv::Rect{0, 0, frame.cols, frame.rows};
        if (!intersection.area()) {
            return;
        }
        cv::Mat graph = frame(intersection);
        graph = graph / 2 + cv::Scalar{127, 127, 127};
        int histxPos = graph.cols - 1;
        double range = std::min(memoryMonitor.getMaxMemTotal() + memoryMonitor.getMaxSwap(),
            (memoryMonitor.getMaxMem() + memoryMonitor.getMaxSwap()) * 1.2);
        if (lastHistory.size() > 1) {
            for (auto memUsageIt = lastHistory.rbegin(); memUsageIt != lastHistory.rend() - 1; ++memUsageIt) {
                constexpr double SWAP_THRESHOLD = 10.0 / 1024; // 10 MiB
                cv::Vec3b color =
                    (memoryMonitor.getMemTotal() * 0.95 > memUsageIt->first) || (memUsageIt->second < SWAP_THRESHOLD) ?
                        cv::Vec3b{0, 255, 255} :
                        cv::Vec3b{0, 0, 255};
                cv::Point right{histxPos,
                    graph.rows - static_cast<int>(graphRectHeight * (memUsageIt->first + memUsageIt->second) / range)};
                cv::Point left{histxPos - sampleStep,
                    graph.rows - static_cast<int>(
                        graphRectHeight * ((memUsageIt + 1)->first + (memUsageIt + 1)->second) / range)};
                cv::line(graph, right, left, color, 2);
                histxPos -= sampleStep;
            }
        }

        cv::Rect border{cv::Point{graphPos, yPos + textGraphSplittingLine},
            cv::Size{graphSize.width, graphSize.height - textGraphSplittingLine}};
        cv::rectangle(frame, {border}, {0, 0, 0});
        if (lastHistory.empty()) {
            strStream.str("Memory");
        } else {
            strStream.str("");
            strStream << std::fixed << std::setprecision(1) << lastHistory.back().first << " + "
                << lastHistory.back().second << " GiB";
        }
        int baseline;
        int textWidth = cv::getTextSize(strStream.str(),
            cv::FONT_HERSHEY_SIMPLEX,
            textGraphSplittingLine * 0.04,
            1,
            &baseline).width;
        cv::putText(graph,
            strStream.str(),
            cv::Point{(graphSize.width - textWidth) / 2, textGraphSplittingLine - 1},
            cv::FONT_HERSHEY_SIMPLEX,
            textGraphSplittingLine * 0.04,
            {0, 35, 35});
    }
}

std::string Presenter::reportMeans() const {
    std::ostringstream collectedDataStream;
    collectedDataStream << std::fixed << std::setprecision(1);
    if (cpuMonitor.getHistorySize() > 1) {
        collectedDataStream << "Mean core utilization: ";
        for (double mean : cpuMonitor.getMeanCpuLoad()) {
            collectedDataStream << mean * 100 << "% ";
        }
        collectedDataStream << '\n';
    }
    if (distributionCpuEnabled) {
        std::vector<double> meanCpuLoad = cpuMonitor.getMeanCpuLoad();
        double mean = std::accumulate(meanCpuLoad.begin(), meanCpuLoad.end(), 0.0) / meanCpuLoad.size();
        collectedDataStream << "Mean CPU utilization: " << mean * 100 << "%\n";
    }
    if (memoryMonitor.getHistorySize() > 1) {
        collectedDataStream << "Memory mean usage: " << memoryMonitor.getMeanMem() << " GiB\n";
        collectedDataStream << "Mean swap usage: " << memoryMonitor.getMeanSwap() << " GiB\n";
    }
    std::string collectedData = collectedDataStream.str();
    // drop last \n because usually it is not expected that printing an object starts a new line
    if (!collectedData.empty()) {
        return collectedData.substr(0, collectedData.size() - 1);
    }
    return collectedData;
}
