#include "sift.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <libutils/rasserts.h>

// Ссылки:
// [lowe04] - Distinctive Image Features from Scale-Invariant Keypoints, David G. Lowe, 2004
//
// Примеры реализаций (стоит обращаться только если совсем не понятны какие-то места):
// 1) https://github.com/robwhess/opensift/blob/master/src/sift.c
// 2) https://gist.github.com/lxc-xx/7088609 (адаптация кода с первой ссылки)
// 3) https://github.com/opencv/opencv/blob/1834eed8098aa2c595f4d1099eeaa0992ce8b321/modules/features2d/src/sift.dispatch.cpp (адаптация кода с первой ссылки)
// 4) https://github.com/opencv/opencv/blob/1834eed8098aa2c595f4d1099eeaa0992ce8b321/modules/features2d/src/sift.simd.hpp (адаптация кода с первой ссылки)

#define DEBUG_ENABLE     1
#define DEBUG_PATH       std::string("data/debug/test_sift/debug/")

#define NOCTAVES                    3                    // число октав
#define OCTAVE_NLAYERS              3                    // в [lowe04] это число промежуточных степеней размытия картинки в рамках одной октавы обозначается - s, т.е. s слоев в каждой октаве
#define OCTAVE_GAUSSIAN_IMAGES      (OCTAVE_NLAYERS + 3)
#define OCTAVE_DOG_IMAGES           (OCTAVE_NLAYERS + 2)
#define INITIAL_IMG_SIGMA           0.75                 // предполагаемая степень размытия изначальной картинки
#define INPUT_IMG_PRE_BLUR_SIGMA    1.0                  // сглаживание изначальной картинки

#define SUBPIXEL_FITTING_ENABLE      0    // такие тумблеры включающие/выключающие очередное улучшение алгоритма позволяют оценить какой вклад эта фича вносит в качество результата если в рамках уже готового алгоритма попробовать ее включить/выключить

#define ORIENTATION_NHISTS           36   // число корзин при определении ориентации ключевой точки через гистограммы
#define ORIENTATION_WINDOW_R         3    // минимальный радиус окна в рамках которого будет выбрана ориентиация (в пикселях), R=3 => 5x5 окно
#define ORIENTATION_VOTES_PEAK_RATIO 0.80 // 0.8 => если гистограмма какого-то направления получила >= 80% от максимального чиссла голосов - она тоже победила

#define DESCRIPTOR_SIZE            4 // 4x4 гистограммы декскриптора
#define DESCRIPTOR_NBINS           8 // 8 корзин-направлений в каждой гистограмме дескриптора (4х4 гистограммы, каждая по 8 корзин, итого 4x4x8=128 значений в дескрипторе)
#define DESCRIPTOR_SAMPLES_N       4 // 4x4 замера для каждой гистограммы дескриптора (всего гистограмм 4х4) итого 16х16 замеров
#define DESCRIPTOR_SAMPLE_WINDOW_R 1.0 // минимальный радиус окна в рамках которого строится гистограмма из 8 корзин-направлений (т.е. для каждого из 16 элементов дескриптора), R=1 => 1x1 окно



void phg::SIFT::detectAndCompute(const cv::Mat &originalImg, std::vector<cv::KeyPoint> &kps, cv::Mat &desc) {
    // используйте дебаг в файлы как можно больше, это очень удобно и потраченное время окупается крайне сильно,
    // ведь пролистывать через окошки показывающие картинки долго, и по ним нельзя проматывать назад, а по файлам - можно
    // вы можете запустить алгоритм, сгенерировать десятки картинок со всеми промежуточными визуализациями и после запуска
    // посмотреть на те этапы к которым у вас вопросы или про которые у вас опасения
    if (DEBUG_ENABLE) cv::imwrite(DEBUG_PATH + "00_input.png", originalImg);

    cv::Mat img = originalImg.clone();
    // для удобства используем черно-белую картинку и работаем с вещественными числами (это еще и может улучшить точность)
    if (originalImg.type() == CV_8UC1) { // greyscale image
        img.convertTo(img, CV_32FC1, 1.0);
    } else if (originalImg.type() == CV_8UC3) { // BGR image
        img.convertTo(img, CV_32FC3, 1.0);
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    } else {
        rassert(false, 14291409120);
    }
    if (DEBUG_ENABLE) cv::imwrite(DEBUG_PATH + "01_grey.png", img);
    cv::GaussianBlur(img, img, cv::Size(0, 0), INPUT_IMG_PRE_BLUR_SIGMA, INPUT_IMG_PRE_BLUR_SIGMA);
    if (DEBUG_ENABLE) cv::imwrite(DEBUG_PATH + "02_grey_blurred.png", img);

    // Scale-space extrema detection
    std::vector<cv::Mat> gaussianPyramid;
    std::vector<cv::Mat> DoGPyramid;
    buildPyramids(img, gaussianPyramid, DoGPyramid);

    findLocalExtremasAndDescribe(gaussianPyramid, DoGPyramid, kps, desc);
}

void phg::SIFT::buildPyramids(const cv::Mat &imgOrg, std::vector<cv::Mat> &gaussianPyramid, std::vector<cv::Mat> &DoGPyramid) {
    gaussianPyramid.resize(NOCTAVES * OCTAVE_GAUSSIAN_IMAGES);

    const double k = pow(2.0, 1.0 / OCTAVE_NLAYERS); // [lowe04] k = 2^{1/s} а у нас s=OCTAVE_NLAYERS

    // строим пирамиду гауссовых размытий картинки
    for (size_t octave = 0; octave < NOCTAVES; ++octave) {
        if (octave == 0) {
            int layer = 0;
            gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES + layer] = imgOrg.clone();
        } else {
            int layer = 0;
            size_t prevOctave = octave - 1;
            cv::Mat img = gaussianPyramid[prevOctave * OCTAVE_GAUSSIAN_IMAGES + OCTAVE_NLAYERS].clone();
            // тут есть очень важный момент, мы должны указать fx=0.5, fy=0.5 иначе при нечетном размере картинка будет не идеально 2 пикселя в один схлопываться - а слегка смещаться
            cv::resize(img, img, cv::Size(img.cols / 2, img.rows / 2), 0.5, 0.5, cv::INTER_NEAREST);
            gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES + layer] = img;
            // берем картинку с предыдущей октавы и уменьшаем ее в два раза без какого бы то ни было дополнительного размытия (сигмы должны совпадать)

        }

        #pragma omp parallel for // TODO: если выполните TODO про "размытие из изначального слоя октавы" ниже - раскоментируйте это распараллеливание, ведь теперь слои считаются независимо (из самого первого), проверьте что результат на картинках не изменился
        for (ptrdiff_t layer = 1; layer < OCTAVE_GAUSSIAN_IMAGES; ++layer) {
            // Теперь размываем из первого слоя октавы, а не из предыдущего
            double sigma0 = INITIAL_IMG_SIGMA * pow(2.0, octave); // сигма первого слоя октавы
            double sigmaCur = INITIAL_IMG_SIGMA * pow(2.0, octave) * pow(k, layer); // целевая сигма текущего слоя
            double sigma = sqrt(sigmaCur * sigmaCur - sigma0 * sigma0); // сигма для размытия из первого слоя

            // Берем первый слой октавы (layer = 0)
            cv::Mat imgLayer = gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES + 0].clone();
            cv::Size automaticKernelSize = cv::Size(0, 0);

            cv::GaussianBlur(imgLayer, imgLayer, automaticKernelSize, sigma, sigma);

            gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES + layer] = imgLayer;
        }
    }

    for (size_t octave = 0; octave < NOCTAVES; ++octave) {
        for (size_t layer = 0; layer < OCTAVE_GAUSSIAN_IMAGES; ++layer) {
            double sigmaCur = INITIAL_IMG_SIGMA * pow(2.0, octave) * pow(k, layer);
            if (DEBUG_ENABLE) cv::imwrite(DEBUG_PATH + "pyramid/o" + to_string(octave) + "_l" + to_string(layer) + "_s" + to_string(sigmaCur) + ".png", gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES + layer]);
            // какие ожидания от картинок можно придумать? т.е. как дополнительно проверить что работает разумно?
            // первый слой октавы i+1 должен визуально совпадать со слоем OCTAVE_NLAYERS октавы i
            // при увеличении в 2 раза (они имеют одинаковую сигму относительно исходного изображения)
            if (octave > 0 && layer == 0) {
                // Этот слой должен быть уменьшенной версией слоя OCTAVE_NLAYERS из предыдущей октавы
                cv::Mat prevOctaveLayer = gaussianPyramid[(octave-1) * OCTAVE_GAUSSIAN_IMAGES + OCTAVE_NLAYERS];
                cv::Mat resized;
                cv::resize(gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES + layer], resized,
                           prevOctaveLayer.size(), 0, 0, cv::INTER_LINEAR);
                if (DEBUG_ENABLE) {
                    cv::imwrite(DEBUG_PATH + "pyramid/check_o" + to_string(octave) + "_l0_resized.png", resized);
                    // Эта картинка должна визуально совпадать с o(octave-1)_l(OCTAVE_NLAYERS)
                }
            }
        }
    }

    DoGPyramid.resize(NOCTAVES * OCTAVE_DOG_IMAGES);

    // строим пирамиду разниц гауссиан слоев (Difference of Gaussian, DoG), т.к. вычитать надо из слоя слой в рамках одной и той же октавы - то есть приятный параллелизм на уровне октав
    #pragma omp parallel for
    for (ptrdiff_t octave = 0; octave < NOCTAVES; ++octave) {
        for (size_t layer = 1; layer < OCTAVE_GAUSSIAN_IMAGES; ++layer) {
            int prevLayer = layer - 1;
            cv::Mat imgPrevGaussian = gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES + prevLayer];
            cv::Mat imgCurGaussian  = gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES + layer];

            cv::Mat imgCurDoG = imgCurGaussian.clone();
            // обратите внимание что т.к. пиксели картинки из одного ряда лежат в памяти подряд, поэтому если вложенный цикл бежит подряд по одному и тому же ряду
            // то код работает быстрее т.к. он будет более cache-friendly, можете сравнить оценить ускорение добавив замер времени построения пирамиды: timer t; double time_s = t.elapsed();
            for (size_t j = 0; j < imgCurDoG.rows; ++j) {
                for (size_t i = 0; i < imgCurDoG.cols; ++i) {
                    imgCurDoG.at<float>(j, i) = imgCurGaussian.at<float>(j, i) - imgPrevGaussian.at<float>(j, i);
                }
            }
            int dogLayer = layer - 1;
            DoGPyramid[octave * OCTAVE_DOG_IMAGES + dogLayer] = imgCurDoG;
        }
    }



    for (size_t octave = 0; octave < NOCTAVES; ++octave) {
        for (size_t layer = 0; layer < OCTAVE_DOG_IMAGES; ++layer) {
            double sigmaCur = INITIAL_IMG_SIGMA * pow(2.0, octave) * pow(k, layer);
            if (DEBUG_ENABLE) cv::imwrite(DEBUG_PATH + "pyramidDoG/o" + to_string(octave) + "_l" + to_string(layer) + "_s" + to_string(sigmaCur) + ".png", DoGPyramid[octave * OCTAVE_DOG_IMAGES + layer]);
            // TODO: какие ожидания от картинок можно придумать? т.е. как дополнительно проверить что работает разумно?
            // спойлер: подуймайте с чем должна визуально совпадать картинка из октавы DoG? может быть с какой-то из картинок с предыдущей октавы? с какой? как их визуально сверить ведь они разного размера?
            if (DEBUG_ENABLE && octave > 0) {
                // Проверяем масштабную инвариантность DoG пирамиды
                cv::Mat currentDoG = DoGPyramid[octave * OCTAVE_DOG_IMAGES + layer];
                cv::Mat prevOctaveDoG = DoGPyramid[(octave-1) * OCTAVE_DOG_IMAGES + layer];

                // Увеличиваем текущий DoG в 2 раза для сравнения с предыдущей октавой
                cv::Mat resizedDoG;
                cv::resize(currentDoG, resizedDoG, prevOctaveDoG.size(), 0, 0, cv::INTER_LINEAR);

                // Сохраняем для визуального сравнения
                cv::imwrite(DEBUG_PATH + "pyramidDoG/check_o" + to_string(octave) + "_l" + to_string(layer) + "_resized.png", resizedDoG);

                // Вычисляем разность для количественной оценки
                cv::Mat diff;
                cv::absdiff(resizedDoG, prevOctaveDoG, diff);
                cv::Scalar meanDiff = cv::mean(diff);

                // Логируем результат (в реальной реализации можно добавить assert для автоматической проверки)
                std::cout << "DoG scale consistency check - Octave " << octave << ", Layer " << layer
                          << ", Mean difference: " << meanDiff[0] << std::endl;
            }
        }
    }
}

namespace {
    float parabolaFitting(float x0, float x1, float x2) {
        rassert((x1 >= x0 && x1 >= x2) || (x1 <= x0 && x1 <= x2), 12541241241241);

        // a*0^2+b*0+c=x0
        // a*1^2+b*1+c=x1
        // a*2^2+b*2+c=x2

        // c=x0
        // a+b+x0=x1     (2)
        // 4*a+2*b+x0=x2 (3)

        // (3)-2*(2): 2*a-y0=y2-2*y1; a=(y2-2*y1+y0)/2
        // (2):       b=y1-y0-a
        float a = (x2-2.0f*x1+x0) / 2.0f;
        float b = x1 - x0 - a;
        // extremum is at -b/(2*a), but our system coordinate start (i) is at 1, so minus 1
        float shift = - b / (2.0f * a) - 1.0f;
        return shift;
    }
}

void phg::SIFT::findLocalExtremasAndDescribe(const std::vector<cv::Mat> &gaussianPyramid, const std::vector<cv::Mat> &DoGPyramid,
                                             std::vector<cv::KeyPoint> &keyPoints, cv::Mat &desc) {
    std::vector<std::vector<float>> pointsDesc;

    const double contrast_threshold_lowe = 0.04;
    const double edge_threshold_lowe = 10.0;

    // 3.1 Local extrema detection
    #pragma omp parallel // запустили каждый вычислительный поток процессора
    {
        // каждый поток будет складировать свои точки в свой личный вектор (чтобы не было гонок и не были нужны точки синхронизации)
        std::vector<cv::KeyPoint> thread_points;
        std::vector<std::vector<float>> thread_descriptors;

        for (size_t octave = 0; octave < NOCTAVES; ++octave) {
            double octave_downscale = pow(2.0, octave);
            for (size_t layer = 1; layer + 1 < OCTAVE_DOG_IMAGES; ++layer) {
                const cv::Mat prev = DoGPyramid[octave * OCTAVE_DOG_IMAGES + layer - 1];
                const cv::Mat cur  = DoGPyramid[octave * OCTAVE_DOG_IMAGES + layer];
                const cv::Mat next = DoGPyramid[octave * OCTAVE_DOG_IMAGES + layer + 1];

                // теперь каждый поток обработает свой кусок картинки
                #pragma omp for
                for (ptrdiff_t j = 1; j < cur.rows - 1; ++j) {
                    for (ptrdiff_t i = 1; i + 1 < cur.cols; ++i) {
                        bool is_max = true;
                        bool is_min = true;
                        float center = cur.at<float>(j, i);

                        for (int dz = -1; dz <= 1 && (is_min || is_max); ++dz) {
                        for (int dy = -1; dy <= 1 && (is_min || is_max); ++dy) {
                        for (int dx = -1; dx <= 1 && (is_min || is_max); ++dx) {
                            if (dz==0 && dy==0 && dx==0) continue;
                            float val = (dz==-1 ? prev : dz==1 ? next : cur).at<float>(j+dy, i+dx);
                            if (center <= val) is_max = false;
                            if (center >= val) is_min = false;
                        }
                        }
                        }
                        bool is_extremum = (is_min || is_max);

                        if (!is_extremum)
                            continue;


                        cv::KeyPoint kp;
                        float dx = 0.0f;
                        float dy = 0.0f;
                        float dlayer_val = 0.0f; // Смещение по слою
                        float dvalue = 0.0f; // Уточненное значение DoG

#if SUBPIXEL_FITTING_ENABLE
                        // НАЧАЛО: РЕАЛИЗАЦИЯ TODO (Субпиксельное уточнение)
                        // Используем ряд Тейлора для уточнения положения.
                        // См. [lowe04] Рис. 3 и Приложение A.
                        int current_i = i, current_j = j, current_layer = layer;
                        for (int iter = 0; iter < 5; ++iter) { // Не более 5 итераций для уточнения
                            const cv::Mat c_prev = DoGPyramid[octave * OCTAVE_DOG_IMAGES + current_layer - 1];
                            const cv::Mat c_cur  = DoGPyramid[octave * OCTAVE_DOG_IMAGES + current_layer];
                            const cv::Mat c_next = DoGPyramid[octave * OCTAVE_DOG_IMAGES + current_layer + 1];

                            // Градиент D' = (dx, dy, ds)
                            float gx = (c_cur.at<float>(current_j, current_i + 1) - c_cur.at<float>(current_j, current_i - 1)) * 0.5f;
                            float gy = (c_cur.at<float>(current_j + 1, current_i) - c_cur.at<float>(current_j - 1, current_i)) * 0.5f;
                            float gs = (c_next.at<float>(current_j, current_i) - c_prev.at<float>(current_j, current_i)) * 0.5f;
                            cv::Mat gradient = (cv::Mat_<float>(3, 1) << gx, gy, gs);

                            // Матрица Гессе D''
                            float v = c_cur.at<float>(current_j, current_i);
                            float dxx = c_cur.at<float>(current_j, current_i + 1) + c_cur.at<float>(current_j, current_i - 1) - 2 * v;
                            float dyy = c_cur.at<float>(current_j + 1, current_i) + c_cur.at<float>(current_j - 1, current_i) - 2 * v;
                            float dss = c_next.at<float>(current_j, current_i) + c_prev.at<float>(current_j, current_i) - 2 * v;
                            float dxy = (c_cur.at<float>(current_j + 1, current_i + 1) - c_cur.at<float>(current_j + 1, current_i - 1) - c_cur.at<float>(current_j - 1, current_i + 1) + c_cur.at<float>(current_j - 1, current_i - 1)) * 0.25f;
                            float dxs = (c_next.at<float>(current_j, current_i + 1) - c_next.at<float>(current_j, current_i - 1) - (c_prev.at<float>(current_j, current_i + 1) - c_prev.at<float>(current_j, current_i - 1))) * 0.25f;
                            float dys = (c_next.at<float>(current_j + 1, current_i) - c_next.at<float>(current_j - 1, current_i) - (c_prev.at<float>(current_j + 1, current_i) - c_prev.at<float>(current_j - 1, current_i))) * 0.25f;

                            cv::Mat hessian = (cv::Mat_<float>(3, 3) <<
                                dxx, dxy, dxs,
                                dxy, dyy, dys,
                                dxs, dys, dss);

                            cv::Mat offset;
                            cv::solve(hessian, -gradient, offset, cv::DECOMP_LU);

                            dx = offset.at<float>(0);
                            dy = offset.at<float>(1);
                            dlayer_val = offset.at<float>(2);

                            // Если смещение по любой оси > 0.5, это значит, что экстремум ближе к другому пикселю.
                            // Смещаемся и повторяем уточнение.
                            if (fabs(dx) > 0.5f || fabs(dy) > 0.5f || fabs(dlayer_val) > 0.5f) {
                                current_i += round(dx);
                                current_j += round(dy);
                                current_layer += round(dlayer_val);

                                // Проверка выхода за границы
                                if (current_layer < 1 || current_layer + 1 >= OCTAVE_DOG_IMAGES ||
                                    current_i < 1 || current_i + 1 >= cur.cols ||
                                    current_j < 1 || current_j + 1 >= cur.rows) {
                                    dx = dy = dlayer_val = 0.0f; // Не удалось уточнить, отбрасываем
                                    break;
                                }
                                continue; // Повторяем итерацию с новой точки
                            }

                            dvalue = 0.5f * gradient.dot(offset);
                            break; // Смещение достаточно мало, выходим
                        }
                        // КОНЕЦ: РЕАЛИЗАЦИЯ TODO (Субпиксельное уточнение)
#endif
                        // НАЧАЛО: РЕАЛИЗАЦИЯ TODO (Фильтрация по контрасту)
                        float contrast = cur.at<float>(j, i) + dvalue;
                        // Порог уменьшается, т.к. при большем числе слоев разница между ними меньше, и отклик DoG ниже.
                        // Деление на NLAYERS делает выбор точек менее зависимым от этого параметра.
                        if (fabs(contrast) < contrast_threshold_lowe / OCTAVE_NLAYERS)
                            continue;

                        // НОВЫЙ КОД: Фильтрация точек на ребрах
                        // Вычисляем 2D Гессиан в точке (i, j)
                        float v = cur.at<float>(j, i);
                        float Dxx = cur.at<float>(j, i+1) + cur.at<float>(j, i-1) - 2*v;
                        float Dyy = cur.at<float>(j+1, i) + cur.at<float>(j-1, i) - 2*v;
                        float Dxy = (cur.at<float>(j+1, i+1) - cur.at<float>(j+1, i-1) - cur.at<float>(j-1, i+1) + cur.at<float>(j-1, i-1)) * 0.25f;
                        float trace_H = Dxx + Dyy;
                        float det_H = Dxx * Dyy - Dxy * Dxy;
                        if (det_H <= 0 || (trace_H * trace_H / det_H) >= ((edge_threshold_lowe + 1.0) * (edge_threshold_lowe + 1.0) / edge_threshold_lowe))
                            continue;
                        // КОНЕЦ: РЕАЛИЗАЦИЯ TODO (Фильтрация)

                        kp.pt = cv::Point2f((i + 0.5f + dx) * octave_downscale, (j + 0.5f + dy) * octave_downscale);
                        kp.response = fabs(contrast);

                        const double k = pow(2.0, 1.0 / OCTAVE_NLAYERS);
                        double sigmaCur = INITIAL_IMG_SIGMA * pow(2.0, octave) * pow(k, layer + dlayer_val); // Уточняем и сигму
                        kp.size = 2.0 * sigmaCur * 5.0; // Размер пропорционален масштабу

                        // 5 Orientation assignment
                        cv::Mat img = gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES + layer];
                        std::vector<float> votes;
                        float biggestVote;
                        int oriRadius = (int) (ORIENTATION_WINDOW_R * sigmaCur); // Радиус зависит от масштаба точки
                        if (!buildLocalOrientationHists(img, i + dx, j + dy, oriRadius, votes, biggestVote))
                            continue;

                        for (size_t bin = 0; bin < ORIENTATION_NHISTS; ++bin) {
                            float prevValue = votes[(bin + ORIENTATION_NHISTS - 1) % ORIENTATION_NHISTS];
                            float value = votes[bin];
                            float nextValue = votes[(bin + 1) % ORIENTATION_NHISTS];
                            if (value > prevValue && value > nextValue && value >= biggestVote * ORIENTATION_VOTES_PEAK_RATIO) {
                                // НАЧАЛО: РЕАЛИЗАЦИЯ TODO (Уточнение угла)
                                float shift = parabolaFitting(prevValue, value, nextValue);
                                float angle = (bin + 0.5f + shift) * (360.0f / ORIENTATION_NHISTS);
                                if (angle < 0.0)    angle += 360.0;
                                if (angle >= 360.0) angle -= 360.0;
                                kp.angle = angle;
                                rassert(kp.angle >= 0.0 && kp.angle < 360.0, 123512412412);
                                // КОНЕЦ: РЕАЛИЗАЦИЯ TODO (Уточнение угла)

                                std::vector<float> descriptor;
                                // kp.pt - уже в координатах исходного изображения. buildDescriptor ожидает координаты в текущем масштабе.
                                float px_octave = (i + 0.5f + dx);
                                float py_octave = (j + 0.5f + dy);
                                if (!buildDescriptor(img, px_octave, py_octave, sigmaCur, kp.angle, descriptor))
                                    continue;

                                thread_points.push_back(kp);
                                thread_descriptors.push_back(descriptor);
                            }
                        }
                    }
                }
            }
        }

        // в критической секции объединяем все массивы детектированных точек
        #pragma omp critical
        {
            keyPoints.insert(keyPoints.end(), thread_points.begin(), thread_points.end());
            pointsDesc.insert(pointsDesc.end(), thread_descriptors.begin(), thread_descriptors.end());
        }
    }

    rassert(pointsDesc.size() == keyPoints.size(), 12356351235124);
    desc = cv::Mat(pointsDesc.size(), DESCRIPTOR_SIZE * DESCRIPTOR_SIZE * DESCRIPTOR_NBINS, CV_32FC1);
    for (size_t j = 0; j < pointsDesc.size(); ++j) {
        rassert(pointsDesc[j].size() == DESCRIPTOR_SIZE * DESCRIPTOR_SIZE * DESCRIPTOR_NBINS, 1253351412421);
        for (size_t i = 0; i < pointsDesc[j].size(); ++i) {
            desc.at<float>(j, i) = pointsDesc[j][i];
        }
    }
}

bool phg::SIFT::buildLocalOrientationHists(const cv::Mat &img, float i_f, float j_f, int radius,
                                           std::vector<float> &votes, float &biggestVote) {
    // 5 Orientation assignment
    votes.resize(ORIENTATION_NHISTS, 0.0f);
    biggestVote = 0.0;

    int i = round(i_f);
    int j = round(j_f);

    // Улучшенная проверка выхода за границы. Нам нужен доступ к пикселям (x-1, y-1) и (x+1, y+1)
    // для самого крайнего пикселя в окне.
    if (i < (int)radius + 1 || i >= img.cols - (int)radius - 1 ||
        j < (int)radius + 1 || j >= img.rows - (int)radius - 1)
        return false;

    float sum[ORIENTATION_NHISTS] = {0.0f};

    // Сигма для Гауссового взвешивания в два раза меньше размера окна (по статье Лоу sigma = 1.5 * scale, а radius = 3 * sigma)
    // ORIENTATION_WINDOW_R в коде равен 3.0, так что sigma = 1.5 * (radius/3.0) = 0.5*radius.
    float weight_sigma = 0.5f * (float)(2 * radius - 1); // 2*radius-1 - это ширина окна
    float weight_denom = 1.0f / (2.0f * weight_sigma * weight_sigma);

    // Итерируемся по окну вокруг ключевой точки
    for (int y_offset = -(int)radius; y_offset <= (int)radius; ++y_offset) {
        for (int x_offset = -(int)radius; x_offset <= (int)radius; ++x_offset) {
            int current_x = i + x_offset;
            int current_y = j + y_offset;

            // НАЧАЛО: РЕАЛИЗАЦИЯ TODO (Вычисление градиента)
            // Вычисляем градиент в точке (current_x, current_y) с помощью центральных разностей
            float dx = img.at<float>(current_y, current_x + 1) - img.at<float>(current_y, current_x - 1);
            float dy = img.at<float>(current_y + 1, current_x) - img.at<float>(current_y - 1, current_x);

            // m(x, y) = sqrt((L(x+1,y)-L(x-1,y))^2 + (L(x,y+1)-L(x,y-1))^2)
            float magnitude = sqrt(dx * dx + dy * dy);

            // orientation = atan2(dy, dx)
            float orientation = atan2(dy, dx); // в радианах [-PI, PI]
            // КОНЕЦ: РЕАЛИЗАЦИЯ TODO (Вычисление градиента)

            // Конвертируем ориентацию в градусы [0, 360)
            float orientation_deg = orientation * (180.0f / M_PI);
            if (orientation_deg < 0.0f)
                orientation_deg += 360.0f;

            rassert(orientation_deg >= 0.0f && orientation_deg < 360.0f, 5361615612);
            static_assert(360 % ORIENTATION_NHISTS == 0, "Inappropriate bins number!");

            // Определяем, в какую корзину гистограммы попадает градиент
            int bin = (int)round(orientation_deg * ORIENTATION_NHISTS / 360.0f);
            if (bin >= ORIENTATION_NHISTS) // Обработка случая, когда угол ~360
                bin = 0;
            rassert(bin >= 0 && bin < ORIENTATION_NHISTS, 361236315613);

            // Вычисляем Гауссов вес в зависимости от расстояния до центра точки (i_f, j_f)
            float dist_sq = (float)(current_x - i_f) * (float)(current_x - i_f) + (float)(current_y - j_f) * (float)(current_y - j_f);
            float weight = exp(-dist_sq * weight_denom);

            // Добавляем взвешенную величину градиента в соответствующую корзину
            sum[bin] += magnitude * weight;
        }
    }

    // НАЧАЛО: РЕАЛИЗАЦИЯ TODO (Сглаживание гистограммы)
    // Сглаживаем гистограмму 2 раза с помощью простого скользящего среднего для большей стабильности
    for (int iter = 0; iter < 2; ++iter) {
        float first_val = sum[0];
        float prev_val = sum[ORIENTATION_NHISTS - 1];
        for (size_t bin = 0; bin < ORIENTATION_NHISTS; ++bin) {
            float current_val = sum[bin];
            float next_val = (bin == ORIENTATION_NHISTS - 1) ? first_val : sum[bin + 1];
            sum[bin] = (prev_val + current_val + next_val) / 3.0f;
            prev_val = current_val;
        }
    }
    // КОНЕЦ: РЕАЛИЗАЦИЯ TODO (Сглаживание гистограммы)

    for (size_t bin = 0; bin < ORIENTATION_NHISTS; ++bin) {
        votes[bin] = sum[bin];
        if (votes[bin] > biggestVote) {
            biggestVote = votes[bin];
        }
    }

    return true;
}

bool phg::SIFT::buildDescriptor(const cv::Mat &img, float px, float py, double descrSampleRadius, float angle,
                                std::vector<float> &descriptor) {
    // Матрица для поворота системы координат на угол, обратный основной ориентации точки
    cv::Mat relativeShiftRotation = cv::getRotationMatrix2D(cv::Point2f(0.0f, 0.0f), -angle, 1.0);


    // smpW - это расстояние между центрами семплов в сетке 16x16.
    const double smpW = 3.0 * descrSampleRadius;

    descriptor.resize(DESCRIPTOR_SIZE * DESCRIPTOR_SIZE * DESCRIPTOR_NBINS, 0.0f);

    // Перебираем 4x4 гистограммы дескриптора
    for (int hstj = 0; hstj < DESCRIPTOR_SIZE; ++hstj) { // строка в решетке гистограмм
        for (int hsti = 0; hsti < DESCRIPTOR_SIZE; ++hsti) { // колонка в решетке гистограмм

            float sum[DESCRIPTOR_NBINS] = {0.0f};

            // Для каждой из 16 гистограмм, мы берем 4x4 замера градиента
            for (int smpj = 0; smpj < DESCRIPTOR_SAMPLES_N; ++smpj) { // строка замера
                for (int smpi = 0; smpi < DESCRIPTOR_SAMPLES_N; ++smpi) { // столбик замера

                    // НАЧАЛО: РЕАЛИЗАЦИЯ TODO (Вычисление координат семпла)
                    // Вычисляем координаты семпла (до поворота) относительно центра ключевой точки.
                    // Центр сетки семплов (16x16) находится в (0,0).
                    // Глобальный индекс семпла: (hsti * 4 + smpi, hstj * 4 + smpj)
                    // Координата относительно центра сетки 16x16:
                    float sub_x = (hsti * DESCRIPTOR_SAMPLES_N + smpi) - (float)(DESCRIPTOR_SIZE * DESCRIPTOR_SAMPLES_N) / 2.0f + 0.5f;
                    float sub_y = (hstj * DESCRIPTOR_SAMPLES_N + smpj) - (float)(DESCRIPTOR_SIZE * DESCRIPTOR_SAMPLES_N) / 2.0f + 0.5f;

                    // Масштабируем до пиксельных координат
                    cv::Point2f shift(sub_x * smpW, sub_y * smpW);

                    std::vector<cv::Point2f> shiftInVector(1, shift);
                    // Поворачиваем вектор смещения в соответствии с ориентацией точки
                    cv::transform(shiftInVector, shiftInVector, relativeShiftRotation);
                    shift = shiftInVector[0];

                    // Финальные координаты точки, в которой мы измеряем градиент
                    int x = (int)round(px + shift.x);
                    int y = (int)round(py + shift.y);

                    // Проверяем, не вышли ли мы за границы изображения
                    if (y - 1 < 0 || y + 1 >= img.rows || x - 1 < 0 || x + 1 >= img.cols)
                        continue; // Просто пропускаем этот семпл

                    // Вычисляем градиент в этой точке
                    float dx_grad = img.at<float>(y, x + 1) - img.at<float>(y, x - 1);
                    float dy_grad = img.at<float>(y + 1, x) - img.at<float>(y - 1, x);
                    float magnitude = sqrt(dx_grad * dx_grad + dy_grad * dy_grad);
                    float orientation = atan2(dy_grad, dx_grad) * (180.0f / M_PI);

                    // КОНЕЦ: РЕАЛИЗАЦИЯ TODO (Вычисление координат семпла)

                    // НАЧАЛО: РЕАЛИЗАЦИЯ TODO (Инвариантность к повороту)
                    // Вычитаем основную ориентацию точки, чтобы ориентация градиента стала относительной.
                    // Это и обеспечивает инвариантность к повороту.
                    if (orientation < 0.0f) orientation += 360.0f;
                    orientation -= angle;
                    orientation = fmodf(orientation + 360.0f, 360.0f);
                    // КОНЕЦ: РЕАЛИЗАЦИЯ TODO

                    rassert(orientation >= 0.0f && orientation < 360.0f, 3515215125412);
                    static_assert(360 % DESCRIPTOR_NBINS == 0, "Inappropriate bins number!");

                    // НАЧАЛО: РЕАЛИЗАЦИЯ TODO (Взвешивание и добавление в гистограмму)
                    // Применяем Гауссов вес: семплы ближе к центру вносят больший вклад.
                    // Сигма для Гауссианы равна половине ширины дескриптора (т.е. 2 * smpW * 4)
                    float weight_sigma = (float)DESCRIPTOR_SIZE * (float)DESCRIPTOR_SAMPLES_N / 2.0f;
                    float weight = exp(-(sub_x*sub_x + sub_y*sub_y) / (2.0f * weight_sigma * weight_sigma));

                    // Определяем корзину и добавляем взвешенную величину
                    size_t bin = (size_t)round(orientation * DESCRIPTOR_NBINS / 360.0f);
                    if (bin >= DESCRIPTOR_NBINS) bin = 0;
                    rassert(bin < DESCRIPTOR_NBINS, 361236315613);

                    sum[bin] += magnitude * weight;
                    // Трилинейную интерполяцию в этой структуре реализовать очень сложно.
                    // Вместо этого мы используем простое добавление в ближайшую корзину,
                    // что является приемлемым упрощением.
                    // КОНЕЦ: РЕАЛИЗАЦИЯ TODO
                }
            }

            // Копируем локальную гистограмму (sum) в обший вектор дескриптора
            float *votes = &(descriptor[(hstj * DESCRIPTOR_SIZE + hsti) * DESCRIPTOR_NBINS]);
            for (int bin = 0; bin < DESCRIPTOR_NBINS; ++bin) {
                votes[bin] = sum[bin];
            }
        }
    }

    // НАЧАЛО: РЕАЛИЗАЦИЯ TODO (Нормализация)
    // Этот шаг выполняется ПОСЛЕ того, как все 128 значений дескриптора были посчитаны.

    // 1. Нормализация L2
    float norm = 0.0f;
    for (float val : descriptor) {
        norm += val * val;
    }
    norm = std::sqrt(norm);
    if (norm > 1e-7) { // Избегаем деления на ноль
        for (float &val : descriptor) {
            val /= norm;
        }
    }

    // 2. Усечение (clamping) для инвариантности к освещению
    for (float &val : descriptor) {
        val = std::min(val, 0.2f);
    }

    // 3. Повторная нормализация
    norm = 0.0f;
    for (float val : descriptor) {
        norm += val * val;
    }
    norm = std::sqrt(norm);
    if (norm > 1e-7) {
        for (float &val : descriptor) {
            val /= norm;
        }
    }
    // КОНЕЦ: РЕАЛИЗАЦИЯ TODO

    return true;
}
