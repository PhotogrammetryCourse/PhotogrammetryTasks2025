#pragma once

#include <vector>

#include <opencv2/core.hpp>



// double contrast_threshold = 0.04; // Порог контрастности
// double edge_threshold = 10.0;     // Порог для отсеивания точек на ребрах
// они такие в статье Лоу
namespace phg {

    class SIFT {
    public:
        // Можете добавить дополнительных параметров со значениями по умолчанию в конструктор если хотите
        SIFT(double contrast_threshold = 0.04, double edge_threshold = 10.0) : contrast_threshold(contrast_threshold) {}

        // Сигнатуру этого метода менять нельзя
        void detectAndCompute(const cv::Mat &originalImg, std::vector<cv::KeyPoint> &kps, cv::Mat &desc);

    protected: // Можете менять внутренние детали реализации включая разбиение на эти методы (это просто набросок):

        void buildPyramids(const cv::Mat &imgOrg, std::vector<cv::Mat> &gaussianPyramid, std::vector<cv::Mat> &DoGPyramid);

        void findLocalExtremasAndDescribe(const std::vector<cv::Mat> &gaussianPyramid, const std::vector<cv::Mat> &DoGPyramid,
                                          std::vector<cv::KeyPoint> &keyPoints, cv::Mat &desc);

        bool buildLocalOrientationHists(const cv::Mat &img, size_t i, size_t j, size_t radius,
                                        std::vector<float> &votes, float &biggestVote);

        bool buildDescriptor(const cv::Mat &img, float px, float py, double descrRadius, float angle,
                             std::vector<float> &descriptor);

        double contrast_threshold;
    };

}





