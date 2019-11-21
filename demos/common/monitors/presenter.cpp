// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cctype>
#include <iomanip>
#include <numeric>

#include "presenter.h"

namespace {
std::set<MonitorType> strKeysToMonitorSet(const std::string& keys) {
    std::set<MonitorType> enabledMonitors;
    for (unsigned char key: keys) {
        switch(std::toupper(key)) {
            case 'C': enabledMonitors.insert(MonitorType::CpuAverage);
                break;
            case 'D': enabledMonitors.insert(MonitorType::DistributionCpu);
                break;
            case 'M': enabledMonitors.insert(MonitorType::Memory);
                break;
            default: throw std::runtime_error("Unknown monitor type");
        }
    }
    return enabledMonitors;
}
}

void meansToOstream(const std::map<MonitorType, std::vector<double>> means, std::ostream& stream) {
    std::ostringstream tmpStream; // create tmp stream to avoid affecting provided stream`s settings
    const auto cpuAverageMean = means.find(MonitorType::CpuAverage);
    if (means.end() != cpuAverageMean) {
        if (cpuAverageMean->second.empty()) {
            tmpStream << "No data collected for CPU utilization\n";
        } else {
            assert (1 == cpuAverageMean->second.size());
            tmpStream << "Mean CPU utilization: " << std::fixed << std::setprecision(1)
                << cpuAverageMean->second.front() * 100 << "%\n";
        }
    }
    const auto distributionCpuMean = means.find(MonitorType::DistributionCpu);
    if (means.end() != distributionCpuMean) {
        if (distributionCpuMean->second.empty()) {
            tmpStream << "No data collected for core utilization\n";
        } else {
            tmpStream << "Mean core utilization: " << std::fixed << std::setprecision(1);
            for (double mean : distributionCpuMean->second) {
                tmpStream << mean * 100 << "% ";
            }
            tmpStream << '\n';
        }
    }
    const auto memoryMean = means.find(MonitorType::Memory);
        if (means.end() != memoryMean) {
            if (memoryMean->second.empty()) {
                tmpStream << "No data collected for memory and swap usage\n";
            } else {
                assert (2 == memoryMean->second.size());
                tmpStream << "Mean memory usage: " << std::fixed << std::setprecision(1) << memoryMean->second.front()
                    << " GiB\n";
                tmpStream << "Mean swap usage: " << std::fixed << std::setprecision(1) << memoryMean->second.back()
                    << " GiB\n";
            }
        }
    stream << tmpStream.str();
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
            strStream{std::ostringstream(std::ios_base::app)} {
    for (MonitorType monitor : enabledMonitors) {
        addRemoveMonitor(monitor);
    }
}

Presenter::Presenter(const std::string& keys, int yPos, cv::Size graphSize, std::size_t historySize) :
    Presenter{strKeysToMonitorSet(keys), yPos, graphSize, historySize} {}

void Presenter::addRemoveMonitor(MonitorType monitor) {
    int sampleStep = std::max(1, static_cast<int>(graphSize.width / historySize));
    unsigned updatedHistorySize = (graphSize.width + sampleStep - 1) / sampleStep; // round up
    updatedHistorySize = std::max(2u, updatedHistorySize);
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

void Presenter::addRemoveMonitor(int key) {
    switch(std::toupper(key)) {
        case 'C': addRemoveMonitor(MonitorType::CpuAverage);
            break;
        case 'D': addRemoveMonitor(MonitorType::DistributionCpu);
            break;
        case 'M': addRemoveMonitor(MonitorType::Memory);
            break;
        case 'H': // show/hide all
            if (0 == cpuMonitor.getHistorySize() && memoryMonitor.getHistorySize() <= 1) {
                addRemoveMonitor(MonitorType::CpuAverage);
                addRemoveMonitor(MonitorType::DistributionCpu);
                addRemoveMonitor(MonitorType::Memory);
            } else {
                cpuMonitor.setHistorySize(0);
                distributionCpuEnabled = false;
                memoryMonitor.setHistorySize(0);
            }
            break;
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
    int sampleStep = std::max(1, static_cast<int>((graphSize.width + historySize - 1) / historySize)); // round up
    unsigned possibleHistorySize = (graphSize.width + sampleStep - 1) / sampleStep; // round up

    if (cpuMonitor.getHistorySize() > 1 && possibleHistorySize > 1 && --numberOfEnabledMonitors >= 0) {
        std::deque<std::vector<double>> lastHistory = cpuMonitor.getLastHistory();
        cv::Mat graph = frame(cv::Rect{cv::Point{graphPos, yPos}, graphSize} & cv::Rect(0, 0, frame.cols, frame.rows));
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
            {255, 0, 0},
            1);
        graphPos += graphSize.width + graphPadding;
    }

    if (distributionCpuEnabled && --numberOfEnabledMonitors >= 0) {
        std::deque<std::vector<double>> lastHistory = cpuMonitor.getLastHistory();
        cv::Mat graph = frame(cv::Rect{cv::Point{graphPos, yPos}, graphSize} & cv::Rect(0, 0, frame.cols, frame.rows));
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
        strStream.str("Core distr");
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
            {0, 255, 0});
        graphPos += graphSize.width + graphPadding;
    }

    if (memoryMonitor.getHistorySize() > 1 && possibleHistorySize > 1 && --numberOfEnabledMonitors >= 0) {
        std::deque<std::pair<double, double>> lastHistory = memoryMonitor.getLastHistory();
        cv::Mat graph = frame(cv::Rect{cv::Point{graphPos, yPos}, graphSize} & cv::Rect(0, 0, frame.cols, frame.rows));
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
            {0, 255, 255});
    }
}

std::map<MonitorType, std::vector<double>> Presenter::getMeans() const {
    std::map<MonitorType, std::vector<double>> means;
    if (cpuMonitor.getHistorySize() > 1) {
        means.emplace(MonitorType::DistributionCpu, cpuMonitor.getMeanCpuLoad());
    }
    if (distributionCpuEnabled) {
        std::vector<double> meanCpuLoad = cpuMonitor.getMeanCpuLoad();
        double mean = std::accumulate(meanCpuLoad.begin(), meanCpuLoad.end(), 0.0) / meanCpuLoad.size();
        means.emplace(MonitorType::CpuAverage, std::vector<double>{mean});
    }
    if (memoryMonitor.getHistorySize() > 1) {
        means.emplace(MonitorType::Memory, std::vector<double>{memoryMonitor.getMeanMem(), memoryMonitor.getMeanSwap()});
    }
    return means;
}
