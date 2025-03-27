#include "triangulation.h"

#include "defines.h"
#include <cstddef>
#include <cstdint>

#include <Eigen/SVD>

// По положениям камер и ключевых точкам определяем точку в трехмерном
// пространстве Задача эквивалентна поиску точки пересечения двух (или более)
// лучей Используем DLT метод, составляем систему уравнений. Система похожа на
// систему для гомографии, там пары уравнений получались из выражений вида x
// (cross) Hx = 0, а здесь будет x (cross) PX = 0 (см. Hartley & Zisserman
// p.312)
cv::Vec4d phg::triangulatePoint(const cv::Matx34d *Ps, const cv::Vec3d *ms,
                                int count) {
  int32_t rows = count * 2;
  int32_t cols = 4;
  Eigen::MatrixXd A(rows, cols);
  for (size_t i = 0; i < rows; ++i) {
    double x = ms[i][0];
    double y = ms[i][1];
    double z = ms[i][2];
    const cv::Matx34d p = Ps[i];
    for (size_t j = 0; j < 4; ++j) {
      A(2 * i, j) = x * p(2, j) - z * p(0, j);
      A(2 * i + 1, j) = y * p(2, j) - z * p(1, j);
    }
  }

  Eigen::JacobiSVD svd(A, Eigen::ComputeFullV);
  Eigen::VectorXd b = svd.matrixV().col(cols - 1);
  cv::Vec4d e(b[0], b[1], b[2], b[3]);
  return e;
}
