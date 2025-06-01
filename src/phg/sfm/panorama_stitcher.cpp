#include "panorama_stitcher.h"

#include <libutils/bbox2.h>

#include <iostream>

#include "homography.h"

namespace {
void dfs(int v, const std::vector<int> &parent, std::vector<bool> &used, std::vector<int> &result) {
    used[v] = true;
    int p = parent[v];
    if (p != -1 && !used[p])
        dfs(p, parent, used, result);
    result.push_back(v);
}

std::vector<int> topological_sort(const std::vector<int> &parent) {
    std::vector<int> result;
    std::vector<bool> used(parent.size(), false);
    for (int i = 0; i < parent.size(); ++i) {
        if (!used[i]) {
            dfs(i, parent, used, result);
        }
    }
    return result;
}
}  // namespace

/*
 * imgs - список картинок
 * parent - список индексов, каждый индекс указывает, к какой картинке должна быть приклеена текущая картинка
 *          этот список образует дерево, корень дерева (картинка, которая ни к кому не приклеивается, приклеиваются только к ней), в данном массиве имеет значение -1
 * homography_builder - функтор, возвращающий гомографию по паре картинок
 * */
cv::Mat phg::stitchPanorama(const std::vector<cv::Mat> &imgs,
                            const std::vector<int> &parent,
                            std::function<cv::Mat(const cv::Mat &, const cv::Mat &)> &homography_builder) {
    const int n_images = imgs.size();

    // склеивание панорамы происходит через приклеивание всех картинок к корню, некоторые приклеиваются не напрямую, а через цепочку других картинок

    // вектор гомографий, для каждой картинки описывает преобразование до корня
    std::vector<cv::Mat> Hs(n_images);
    {
        std::vector<int> order = topological_sort(parent);
        if (n_images > 0)
            Hs[order[0]] = cv::Mat::eye(3, 3, CV_64FC1);
        for (int i = 1; i < n_images; ++i) {
            int cur_img = order[i];
            const cv::Mat homography_to_parent = homography_builder(imgs[cur_img], imgs[parent[cur_img]]);
            Hs[cur_img] = Hs[parent[cur_img]] * homography_to_parent;
        }
    }

    bbox2<double, cv::Point2d> bbox;
    for (int i = 0; i < n_images; ++i) {
        double w = imgs[i].cols;
        double h = imgs[i].rows;
        bbox.grow(phg::transformPoint(cv::Point2d(0.0, 0.0), Hs[i]));
        bbox.grow(phg::transformPoint(cv::Point2d(w, 0.0), Hs[i]));
        bbox.grow(phg::transformPoint(cv::Point2d(w, h), Hs[i]));
        bbox.grow(phg::transformPoint(cv::Point2d(0, h), Hs[i]));
    }

    std::cout << "bbox: " << bbox.max() << ", " << bbox.min() << std::endl;

    int result_width = bbox.width() + 1;
    int result_height = bbox.height() + 1;

    cv::Mat result = cv::Mat::zeros(result_height, result_width, CV_8UC3);

    // из-за растяжения пикселей при использовании прямой матрицы гомографии после отображения между пикселями остается пустое пространство
    // лучше использовать обратную и для каждого пикселя на итоговвой картинке проверять, с какой картинки он может получить цвет
    // тогда в некоторых пикселях цвет будет дублироваться, но изображение будет непрерывным
    //        for (int i = 0; i < n_images; ++i) {
    //            for (int y = 0; y < imgs[i].rows; ++y) {
    //                for (int x = 0; x < imgs[i].cols; ++x) {
    //                    cv::Vec3b color = imgs[i].at<cv::Vec3b>(y, x);
    //
    //                    cv::Point2d pt_dst = applyH(cv::Point2d(x, y), Hs[i]) - bbox.min();
    //                    int y_dst = std::max(0, std::min((int) std::round(pt_dst.y), result_height - 1));
    //                    int x_dst = std::max(0, std::min((int) std::round(pt_dst.x), result_width - 1));
    //
    //                    result.at<cv::Vec3b>(y_dst, x_dst) = color;
    //                }
    //            }
    //        }

    std::vector<cv::Mat> Hs_inv;
    std::transform(Hs.begin(), Hs.end(), std::back_inserter(Hs_inv), [&](const cv::Mat &H) { return H.inv(); });

#pragma omp parallel for
    for (int y = 0; y < result_height; ++y) {
        for (int x = 0; x < result_width; ++x) {
            cv::Point2d pt_dst(x, y);

            // test all images, pick first
            for (int i = 0; i < n_images; ++i) {
                cv::Point2d pt_src = phg::transformPoint(pt_dst + bbox.min(), Hs_inv[i]);

                int x_src = std::round(pt_src.x);
                int y_src = std::round(pt_src.y);

                if (x_src >= 0 && x_src < imgs[i].cols && y_src >= 0 && y_src < imgs[i].rows) {
                    result.at<cv::Vec3b>(y, x) = imgs[i].at<cv::Vec3b>(y_src, x_src);
                    break;
                }
            }
        }
    }

    return result;
}
