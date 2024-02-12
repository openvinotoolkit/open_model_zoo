// Copyright (C) 2021-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <tuple>
#include <vector>
#include "geodist.hpp"

typedef std::pair<cv::Point2d, cv::Point2d> Line2d;

std::pair<double, double> getLineComponent(const cv::Point2d& p1, const cv::Point2d& p2) {
    double run = p2.x - p1.x;
    double rise = p2.y - p1.y;
    double a = 0;

    if (std::fabs(run) < 0.0000001) {
        double sign = run < 0 ? -1.0 : 1.0;
        a = rise / (sign * 0.0000001);
    } else {
        a = rise / run;
    }

    double k = p1.y - (a * p1.x);

    return std::make_pair(a, k);
}

Line2d getLine(const cv::Point2d& A, const cv::Point2d& B) {
    return std::make_pair(A, B);
}

double getX(double y, double a, double k) {
    return (y - k) / a;
}

double getY(double x, double a, double k) {
    return a * x + k;
}

double getDistance(const cv::Point2d& A, const cv::Point2d& B) {
    return sqrt(pow((A.x - B.x), 2) + pow((A.y - B.y), 2));
}

double lineLength(const Line2d& line) {
    return sqrt(pow((line.first.x - line.second.x), 2) + pow((line.first.y - line.second.y), 2));
}

cv::Point2d lineCentroid(const Line2d& line) {
    return cv::Point2d((line.first.x + line.second.x) / 2, (line.first.y + line.second.y) / 2);
}

std::vector<Line2d> cut(const Line2d& line, double distance) {
    double llen = getDistance(line.first, line.second);

    std::vector<Line2d> ret;

    if (distance <= 0 || distance >= llen) {
        ret.push_back(line);
        return ret;
    }

    double a = distance / llen;
    cv::Point2d C(line.first.x + a * (line.second.x - line.first.x), line.first.y + a * (line.second.y - line.first.y));

    ret.push_back(Line2d(line.first, C));
    ret.push_back(Line2d(C, line.second));

    return ret;
}

cv::Point2d lineIntersection(const Line2d& l1, const Line2d& l2) {
    double x1 = l1.first.x;
    double x2 = l1.second.x;
    double x3 = l2.first.x;
    double x4 = l2.second.x;
    double y1 = l1.first.y;
    double y2 = l1.second.y;
    double y3 = l2.first.y;
    double y4 = l2.second.y;

    double d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
    double t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / d;

    return cv::Point2d(x1 + t * (x2 - x1), y1 + t * (y2 - y1));
}

bool lineContainsPoint(const Line2d& l, const cv::Point2d& p) {
    const cv::Point2d& a = l.first;
    const cv::Point2d& b = l.second;

    if (std::isnan(a.x) || std::isnan(a.y) || std::isnan(b.x) || std::isnan(b.y) ||
        std::isnan(p.x) || std::isnan(p.x)) {
        return false;
    }

    double crossproduct = (b.x - a.x) * (p.y - a.y) - (p.x - a.x) * (b.y - a.y);
    double dotproduct = (p.x - a.x) * (b.x - a.x) + (p.y - a.y) * (b.y - a.y);

    if (std::fabs(crossproduct/dotproduct) > 0.0000001) {
        return false;
    }

    if (dotproduct < 0) {
        return false;
    }

    double sqlen = (b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y);
    if (dotproduct > sqlen) {
        return false;
    }

    return true;
}

// euclidean, alert, distance
std::tuple<bool, bool, double> euclideanDistance(const Line2d& AB, const Line2d& CD,
                                                  unsigned minIter, double coef) {
    cv::Point2d Z1 = lineCentroid(AB);
    cv::Point2d Z2 = lineCentroid(CD);

    double minDist = (lineLength(AB) * minIter) * 0.8;
    double distance = getDistance(Z1, Z2) * coef;
    if (distance < lineLength(AB)) {
        return std::make_tuple(true, false, distance);
    }

    bool alert = distance < minDist ? true : false;

    return std::make_tuple(true, alert, distance);
}

double getDistance(const cv::Point2d& PF, const cv::Point2d& E, const cv::Point2d& F,
                    cv::Point2d& za, cv::Point2d& zb, const cv::Point2d& zLimitA, const cv::Point2d& zLimitB,
                    double inf1Dir, unsigned initIter = 1, double coef = 1) {
    double _inf = inf1Dir;
    double inf = inf1Dir * -1;
    bool zaOverZLimit = false;
    Line2d PFE = getLine(PF, E);
    Line2d PFF = getLine(PF, F);
    double lenEF = getDistance(E, F);
    unsigned cnt = initIter;

    while (!zaOverZLimit) {
        if (inf > 0) {
            double lenZbF = getDistance(zb, F);
            cv::Point2d fProj(F.x, F.y + lenZbF);
            cv::Point2d eProj(E.x, F.y + lenZbF + lenEF);

            double projA, projK;
            std::tie(projA, projK) = getLineComponent(eProj, fProj);
            cv::Point2d AUX(getX(F.y, projA, projK), F.y);

            double auxA, auxK;
            std::tie(auxA, auxK) = getLineComponent(zb, AUX);
            Line2d AUXinf = getLine(cv::Point2d(_inf, getY(_inf, auxA, auxK)), AUX);
            cv::Point2d zAUXa = lineIntersection(PFE, AUXinf);

            if (zAUXa.y < zLimitA.y) {
                zaOverZLimit = true;
            } else {
                cnt += 1;
                za = zAUXa;
                Line2d zbAUX = getLine(za, cv::Point2d(inf, za.y));
                zb = lineIntersection(PFF, zbAUX);
            }
        } else {
            double lenZaE = getDistance(za, E);
            cv::Point2d fProj(F.x, F.y + lenZaE + lenEF);
            cv::Point2d eProj(E.x, F.y + lenZaE);

            double projA, projK;
            std::tie(projA, projK) = getLineComponent(eProj, fProj);
            cv::Point2d AUX(getX(E.y, projA, projK), E.y);

            double auxA, auxK;
            std::tie(auxA, auxK) = getLineComponent(za, AUX);
            Line2d AUXinf = getLine(cv::Point2d(_inf, getY(_inf, auxA, auxK)), AUX);
            cv::Point2d zAUXb = lineIntersection(PFF, AUXinf);

            if (zAUXb.y < zLimitB.y) {
                zaOverZLimit = true;
            } else {
                cnt += 1;
                zb = zAUXb;
                Line2d zaAUX = getLine(zb, cv::Point2d(inf, zb.y));
                za = lineIntersection(PFE, zaAUX);
            }
        }
    }

    return cnt * coef;
}

// euclidean, alert, distance
std::tuple<bool, bool, double> socialDistance(std::tuple<int, int>& frameShape,
                                               cv::Point2d& A, cv::Point2d& B,
                                               cv::Point2d& C, cv::Point2d& D,
                                               unsigned minIter, double minW, double maxW) {
    double h, w;
    std::tie(h, w) = frameShape;
    Line2d AB = getLine(A, B);
    Line2d CD = getLine(C, D);

    double COEF = 1;

    double minx = A.x < C.x ? A.x : C.x;
    double maxx = B.x > D.x ? B.x : D.x;

    if (minW * 1.8 <= maxW && minx <= w * .1) {
        return std::make_tuple(true, false, 0);
    }

    bool inBorder = minx < w * .3 || maxx > w - (w * .3) ? true : false;
    double thr = inBorder ? .1 : .01;
    if (std::fabs(lineLength(CD) - lineLength(AB)) <= thr || std::fabs(C.y - A.y) <= h * .01) {
        double p = ((lineLength(CD) + lineLength(AB) / 2) - minW) / maxW;
        if (p < .3) {
            COEF = 1.0 + (1 - p);
        }
        return euclideanDistance(AB, CD, minIter, COEF);
    }

    // Calculation of ordered slope to the origin of line BD
    double bdA, bdK, acA, acK;
    std::tie(bdA, bdK) = getLineComponent(B, D);
    std::tie(acA, acK) = getLineComponent(A, C);

    double bdinf = std::lround(B.x) <= std::lround(D.x) ? -9999999999. : 9999999999.;
    Line2d BDinf = getLine(D, cv::Point2d(bdinf, getY(bdinf, bdA, bdK)));

    double acinf = std::lround(A.x) <= std::lround(C.x) ? -9999999999. : 9999999999.;
    Line2d ACinf = getLine(C, cv::Point2d(acinf, getY(acinf, acA, acK)));

    // Vanishing point
    cv::Point2d PF = lineIntersection(BDinf, ACinf);

    if (!lineContainsPoint(BDinf, PF)) {
        double p = ((lineLength(CD) + lineLength(AB) / 2) - minW) / maxW;
        if (p < .3) {
            COEF = 1.0 + (1 - p);
        }
        return euclideanDistance(AB, CD, minIter, COEF);
    }

    cv::Point2d E(getX(h, acA, acK), h);
    cv::Point2d F(getX(h, bdA, bdK), h);

    unsigned initIter = 1;
    if (E.y - C.y < 1) {
        if (bdinf > 0) {
            Line2d EPF = getLine(E, PF);
            const cv::Point2d newC = cut(EPF, lineLength(CD))[0].second;
            if (A.y < newC.y) {
                initIter += 1;
                C = cv::Point2d(newC.x, newC.y);
            } else {
                return std::make_tuple(false, false, initIter);
            }
        } else {
            Line2d FPF = getLine(F, PF);
            const cv::Point2d newD = cut(FPF, lineLength(CD))[0].second;
            if (B.y < newD.y) {
                initIter += 1;
                D = cv::Point2d(newD.x, newD.y);
            } else {
                return std::make_tuple(false, false, initIter);
            }
        }
    }

    if (bdinf > 0) {
        Line2d Z = getLine(F, PF);
        double frac = lineLength(Z) / 2;

        auto l = cut(Z, frac);
        Line2d& l2 = l[1];

        if (lineContainsPoint(l2, B)) {
            const cv::Point2d& med = l2.first;

            double medB = getDistance(med, B);
            double medPF = getDistance(med, PF);
            double dist = medB / medPF;
            COEF = exp(1 + dist);
        }
    } else {
        Line2d Z = getLine(E, PF);
        double frac = lineLength(Z) / 2;

        auto l = cut(Z, frac);
        Line2d& l2 = l[1];

        if (lineContainsPoint(l2, A)) {
            const cv::Point2d& med = l2.first;

            double medA = getDistance(med, A);
            double medPF = getDistance(med, PF);
            double dist = medA / medPF;
            COEF = exp(1 + dist);
        }
    }

    double cnt = getDistance(PF, E, F, C, D, A, B, bdinf, initIter, COEF);
    bool alert = cnt >= minIter ? false : true;
    return std::make_tuple(false, alert, cnt);
}
