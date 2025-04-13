#include "triangulation.h"

#include "libutils/rasserts.h"

#include <Eigen/SVD>
#include <Eigen/src/SVD/JacobiSVD.h>

// По положениям камер и ключевых точкам определяем точку в трехмерном пространстве
// Задача эквивалентна поиску точки пересечения двух (или более) лучей
// Используем DLT метод, составляем систему уравнений. Система похожа на систему для гомографии, там пары уравнений получались из выражений вида x (cross) Hx = 0, а здесь будет x (cross) PX = 0
// (см. Hartley & Zisserman p.312)
cv::Vec4d phg::triangulatePoint(const cv::Matx34d *Ps, const cv::Vec3d *ms, int count)
{
    Eigen::MatrixXd A(2 * count, 4);
    for (int i = 0; i < count; i++) {
        double x = ms[i][0], y = ms[i][1], z = ms[i][2];
        rassert(abs(z - 1) < 1e-3 , 99232157);

        auto [row1] = Ps[i].row(2) * x - Ps[i].row(0) * z;
        auto [row2] = Ps[i].row(2) * y - Ps[i].row(1) * z;

        A.row(2*i) << row1[0], row1[1], row1[2], row1[3];
        A.row(2*i+1) << row2[0], row2[1], row2[2], row2[3];
    }

    auto null_space = Eigen::JacobiSVD(A, Eigen::ComputeFullV).matrixV().col(3);
    return {null_space[0], null_space[1], null_space[2], null_space[3]};
}
