// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <tuple>
#include <opencv2/core/core.hpp>

#include <vector>


typedef std::pair<cv::Point2d, cv::Point2d> Line2d;

std::pair<double, double> get_line_component(const cv::Point2d &p1, const cv::Point2d &p2) {
    double run = p2.x - p1.x;
    double rise = p2.y - p1.y;
    double a = 0;

    if (std::fabs(run) < 0.0000001) {
        double sign = (run > 0) - (run < 0);
        a = rise / (sign * 0.0000001);
    } else {
        a = rise / run;
    }

    double k = p1.y - (a * p1.x);

    return std::make_pair(a, k);
}

Line2d get_line(const cv::Point2d &A, const cv::Point2d &B) {
    return std::make_pair(A, B);
}

double get_x(double y, double a, double k) {
    return (y - k) / a;
}

double get_y(double x, double a, double k) {
    return a * x + k;
}

double get_distance(const cv::Point2d& A, const cv::Point2d& B) {
    return sqrt(pow((A.x - B.x), 2) + pow((A.y - B.y), 2));
}

double line_length(const Line2d& line) {
    return sqrt(pow((line.first.x - line.second.x), 2) + pow((line.first.y - line.second.y), 2));
}

cv::Point2d line_centroid(const Line2d& line) {
    return cv::Point2d((line.first.x + line.second.x) / 2, (line.first.y + line.second.y) / 2);
}

std::vector<Line2d> cut(const Line2d &line, double distance) {
    double llen = get_distance(line.first, line.second);

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

cv::Point2d line_intersection(const Line2d& l1, const Line2d& l2) {
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

bool line_contains_point(const Line2d& l, const cv::Point2d& p) {
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

std::tuple<int, int, int, int> get_crop(std::tuple<int, int, int, int> a, std::tuple<int, int, int, int> b) {
    int axmin, aymin, axmax, aymax;
    int bxmin, bymin, bxmax, bymax;
    int cxmin, cymin, cxmax, cymax;

    std::tie(axmin, aymin, axmax, aymax) = a;
    std::tie(bxmin, bymin, bxmax, bymax) = b;

    cxmin = axmin < bxmin ? axmin : bxmin;
    cymin = aymin < bymin ? aymin : bymin;
    cxmax = axmax > bxmax ? axmax : bxmax;
    cymax = aymax > bymax ? aymax : bymax;

    return std::make_tuple(cxmin, cymin, cxmax, cymax);
}

// euclidean, alert, distance
std::tuple<bool, bool, double> euclidean_distance(const Line2d &AB, const Line2d &CD,
                                                  unsigned min_iter, double coef) {
    cv::Point2d Z1 = line_centroid(AB);
    cv::Point2d Z2 = line_centroid(CD);

    double min_dist = (line_length(AB) * min_iter) * 0.8;
    double distance = get_distance(Z1, Z2) * coef;
    if (distance < line_length(AB)) {
        return std::make_tuple(true, false, distance);
    }

    bool alert = distance < min_dist ? true : false;

    return std::make_tuple(true, alert, distance);
}

double get_distance(const cv::Point2d &PF, const cv::Point2d &E, const cv::Point2d &F,
                    cv::Point2d &Za, cv::Point2d &Zb, const cv::Point2d &Zlimit_a, const cv::Point2d &Zlimit_b,
                    double inf1_dir, unsigned init_iter = 1, double coef = 1) {
    double _inf = inf1_dir;
    double inf = inf1_dir * -1;
    bool Za_over_Zlimit = false;
    Line2d PFE = get_line(PF, E);
    Line2d PFF = get_line(PF, F);
    double len_EF = get_distance(E, F);
    unsigned cnt = init_iter;

    while (!Za_over_Zlimit) {
        if (inf > 0) {
            // inf = 9999
            // _inf = -9999
            double len_ZbF = get_distance(Zb, F);
            cv::Point2d F_proj(F.x, F.y + len_ZbF);
            cv::Point2d E_proj(E.x, F.y + len_ZbF + len_EF);

            double proj_a, proj_k;
            std::tie(proj_a, proj_k) = get_line_component(E_proj, F_proj);
            cv::Point2d AUX(get_x(F.y, proj_a, proj_k), F.y);

            double aux_a, aux_k;
            std::tie(aux_a, aux_k) = get_line_component(Zb, AUX);
            Line2d AUXinf = get_line(cv::Point2d(_inf, get_y(_inf, aux_a, aux_k)), AUX);
            cv::Point2d Zaux_a = line_intersection(PFE, AUXinf);

            if (Zaux_a.y < Zlimit_a.y) {
                Za_over_Zlimit = true;
            } else {
                cnt += 1;
                Za = Zaux_a;
                Line2d Zb_aux = get_line(Za, cv::Point2d(inf, Za.y));
                Zb = line_intersection(PFF, Zb_aux);
            }
        } else {
            // inf = - 9999
            // _inf = 9999
            double len_ZaE = get_distance(Za, E);
            cv::Point2d F_proj(F.x, F.y + len_ZaE + len_EF);
            cv::Point2d E_proj(E.x, F.y + len_ZaE);

            double proj_a, proj_k;
            std::tie(proj_a, proj_k) = get_line_component(E_proj, F_proj);
            cv::Point2d AUX(get_x(E.y, proj_a, proj_k), E.y);

            double aux_a, aux_k;
            std::tie(aux_a, aux_k) = get_line_component(Za, AUX);
            Line2d AUXinf = get_line(cv::Point2d(_inf, get_y(_inf, aux_a, aux_k)), AUX);
            cv::Point2d Zaux_b = line_intersection(PFF, AUXinf);

            if (Zaux_b.y < Zlimit_b.y) {
                Za_over_Zlimit = true;
            } else {
                cnt += 1;
                Zb = Zaux_b;
                Line2d Za_aux = get_line(Zb, cv::Point2d(inf, Zb.y));
                Za = line_intersection(PFE, Za_aux);
            }
        }
    }

    cnt = cnt * coef;
    return cnt;
}

// euclidean, alert, distance
std::tuple<bool, bool, double> social_distance(std::tuple<int, int> &frame_shape,
                                               std::tuple<int, int> &a, std::tuple<int, int> &b,
                                               std::tuple<int, int> &c, std::tuple<int, int> &d,
                                               unsigned min_iter = 3, double min_w = 0, double max_w = 0) {
    double h, w;
    std::tie(h, w) = frame_shape;
    cv::Point2d A(std::get<0>(a), std::get<1>(a));
    cv::Point2d B(std::get<0>(b), std::get<1>(b));
    cv::Point2d C(std::get<0>(c), std::get<1>(c));
    cv::Point2d D(std::get<0>(d), std::get<1>(d));
    Line2d AB = get_line(A, B);
    Line2d CD = get_line(C, D);

    double COEF = 1;

    double minx = A.x < C.x ? A.x : C.x;
    double maxx = B.x > D.x ? B.x : D.x;

    if (min_w * 1.8 <= max_w && minx <= w * .1) {
        return std::make_tuple(true, false, 0);
    }

    bool in_border = minx < w * .3 || maxx > w - (w * .3) ? true : false;
    double thr = in_border ? .1 : .01;
    if (std::fabs(line_length(CD) - line_length(AB)) <= thr || std::fabs(C.y - A.y) <= h * .01) {
        double p = ((line_length(CD) + line_length(AB) / 2) - min_w) / max_w;
        if (p < .3) {
            COEF = 1.0 + (1 - p);
        }
        return euclidean_distance(AB, CD, min_iter, COEF);
    }

    // Calculation of ordered slope to the origin of line BD
    double bd_a, bd_k, ac_a, ac_k;
    std::tie(bd_a, bd_k) = get_line_component(B, D);
    std::tie(ac_a, ac_k) = get_line_component(A, C);

    double bdinf = B.x < D.x ? -9999999999 : 9999999999;
    Line2d BDinf = get_line(D, cv::Point2d(bdinf, get_y(bdinf, bd_a, bd_k)));

    double acinf = A.x < C.x ? -9999999999 : 9999999999;
    Line2d ACinf = get_line(C, cv::Point2d(acinf, get_y(acinf, ac_a, ac_k)));

    // Vanishing point
    cv::Point2d PF = line_intersection(BDinf, ACinf);

    if (!line_contains_point(BDinf, PF)) {
        double p = ((line_length(CD) + line_length(AB) / 2) - min_w) / max_w;
        if (p < .3) {
            COEF = 1.0 + (1 - p);
        }
        return euclidean_distance(AB, CD, min_iter, COEF);
    }

    cv::Point2d E(get_x(h, ac_a, ac_k), h);
    cv::Point2d F(get_x(h, bd_a, bd_k), h);

    unsigned init_iter = 1;
    if (E.y - C.y < 1) {
        if (bdinf > 0) {
            Line2d EPF = get_line(E, PF);
            const cv::Point2d& new_c = cut(EPF, line_length(CD))[0].second;
            if (A.y < new_c.y) {
                init_iter += 1;
                C = cv::Point2d(new_c.x, new_c.y);
            } else {
                return std::make_tuple(false, false, init_iter);
            }
        } else {
            Line2d FPF = get_line(F, PF);
            const cv::Point2d& new_d = cut(FPF, line_length(CD))[0].second;
            if (B.y < new_d.y) {
                init_iter += 1;
                D = cv::Point2d(new_d.x, new_d.y);
            } else {
                return std::make_tuple(false, false, init_iter);
            }
        }
    }

    if (bdinf > 0) {
        Line2d Z = get_line(F, PF);
        double frac = line_length(Z) / 2;

        auto l = cut(Z, frac);
        Line2d& l2 = l[1];

        if (line_contains_point(l2, B)) {
            const cv::Point2d& med = l2.first;

            double med_b = get_distance(med, B);
            double med_pf = get_distance(med, PF);
            double dist = med_b / med_pf;
            COEF = exp(1 + dist);
        }
    } else {
        Line2d Z = get_line(E, PF);
        double frac = line_length(Z) / 2;

        auto l = cut(Z, frac);
        Line2d& l2 = l[1];

        if (line_contains_point(l2, A)) {
            const cv::Point2d& med = l2.first;

            double med_a = get_distance(med, A);
            double med_pf = get_distance(med, PF);
            double dist = med_a / med_pf;
            COEF = exp(1 + dist);
        }
    }

    double cnt = get_distance(PF, E, F, C, D, A, B, bdinf, init_iter, COEF);
    bool alert = cnt >= min_iter ? false : true;
    return std::make_tuple(false, alert, cnt);
}
