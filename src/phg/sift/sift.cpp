#include "sift.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

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

#define SUBPIXEL_FITTING_ENABLE      1    // такие тумблеры включающие/выключающие очередное улучшение алгоритма позволяют оценить какой вклад эта фича вносит в качество результата если в рамках уже готового алгоритма попробовать ее включить/выключить
#define SUBBIN_FITTING_ENABLE        1
#define CONTRAST_FILTERING_ENABLE    1
#define PARALLEL_ENABLE              0
#define PARALLEL_DOG_ENABLE          0

#define ORIENTATION_NHISTS           36   // число корзин при определении ориентации ключевой точки через гистограммы
#define ORIENTATION_WINDOW_R         3    // минимальный радиус окна в рамках которого будет выбрана ориентиация (в пикселях), R=3 => 5x5 окно
#define ORIENTATION_VOTES_PEAK_RATIO 0.80 // 0.8 => если гистограмма какого-то направления получила >= 80% от максимального чиссла голосов - она тоже победила

#define DESCRIPTOR_SIZE            4 // 4x4 гистограммы декскриптора
#define DESCRIPTOR_NBINS           8 // 8 корзин-направлений в каждой гистограмме дескриптора (4х4 гистограммы, каждая по 8 корзин, итого 4x4x8=128 значений в дескрипторе)
#define DESCRIPTOR_SAMPLES_N       4 // 4x4 замера для каждой гистограммы дескриптора (всего гистограмм 4х4) итого 16х16 замеров
#define DESCRIPTOR_SAMPLE_WINDOW_R 1.0 // минимальный радиус окна в рамках которого строится гистограмма из 8 корзин-направлений (т.е. для каждого из 16 элементов дескриптора), R=1 => 1x1 окно

#define HIST_SMOOTHING_FACTOR      0.2

void phg::SIFT::detectAndCompute(const cv::Mat &originalImg, std::vector<cv::KeyPoint> &kps, cv::Mat &desc) {
    // используйте дебаг в файлы как можно больше, это очень удобно и потраченное время окупается крайне сильно,
    // ведь пролистывать через окошки показывающие картинки долго, и по ним нельзя проматывать назад, а по файлам - можно
    // вы можете запустить алгоритм, сгенерировать десятки картинок со всеми промежуточными визуализациями и после запуска
    // посмотреть на те этапы к которым у вас вопросы или про которые у вас опасения

    // Если бы я только знал как должны выглядеть картинки которые выдает программа, то может быть мне помогло бы :sigh:
    // Но прикола с октавами я не понял
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
            // берем картинку с предыдущей октавы и уменьшаем ее в два раза без какого бы то ни было дополнительного размытия (сигмы должны совпадать)
            cv::Mat img = gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES + layer - 3].clone();
            // тут есть очень важный момент, мы должны указать fx=0.5, fy=0.5 иначе при нечетном размере картинка будет не идеально 2 пикселя в один схлопываться - а слегка смещаться
            cv::resize(img, img, cv::Size( img.cols / 2, img.rows / 2), 0.5, 0.5, cv::INTER_NEAREST);
            gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES + layer] = img;
        }

#if PARALLEL_ENABLE && PARALLEL_DOG_ENABLE
        #pragma omp parallel for
#endif
        for (ptrdiff_t layer = 1; layer < OCTAVE_GAUSSIAN_IMAGES; ++layer) {
            size_t prevLayer = layer - 1;
            // TODONE: переделайте это добавочное размытие с варианта "размываем предыдущий слой" на вариант "размываем самый первый слой октавы до степени размытия сигмы нашего текущего слоя"
            // проверьте - картинки отладочного вывода выглядят один-в-один до/после? (посмотрите на них туда-сюда быстро мигая)
            cv::Size automaticKernelSize = cv::Size(0, 0);
#if !PARALLEL_DOG_ENABLE
            // если есть два последовательных гауссовых размытия с sigma1 и sigma2, то результат будет с sigma12=sqrt(sigma1^2 + sigma2^2) => sigma2=sqrt(sigma12^2-sigma1^2)
            double sigmaPrev = INITIAL_IMG_SIGMA * pow(2.0, octave) * pow(k, prevLayer); // sigma1  - сигма до которой дошла картинка на предыдущем слое
            // Q: посмотрите внимательно на формулу выше и решите как по мнению этой формулы соотносится сигма у первого А-слоя i-ой октавы
            //    и сигма у одного из последних слоев Б предыдущей (i-1)-ой октавы из которого этот слой А был получен?
            // Note: в процессе кода я разнес формулу.
            // A: = (2^i     * k^1) / (2^(i-1) * k^(OGI - 1)) = 2 / k^(OGI-1) = 2 / 2^((OGI-1)/(ONL)) = 2 / 2 ^((ONL-2)/(ONL)) = 2^(1 - (ONL-2)/ONL) = 2^(2/ONL) = k^2
            //      Если взять вместо последнего слоя - предпоследний, то получим отношение k,
            //      Пред-предпоследний - единицу

            // Q: а как чисто идейно должны бы соотноситься сигмы размытия у двух картинок если картинка А была получена из картинки Б простым уменьшением в 2 раза?
            // A: На картинке Б радиус размытия в 2 раза меньше чем на А даст примерно такой же результат. Примерно - потому что при уменьшении картинки теряем детали,
            //    которые могли бы внести свой вклад в цвет пикселя. Но сигма не радиус, а что-то другое - (среднеквадратичное отклонение?).
            //    Поэтому при уменьшении радиуса в 2 раза - сигма должна быть в 4 раза меньше

            //    Есть некоторое подозрение, что я ничего не понял в октавах и сигмах.
            //    То есть наверху при подготовке нулевого слоя каждой октавы сказано что сигмы должны совпадать, а тут я пишу что-то что с жтим не совпадает,
            //    Разница отношений то 4, то откуда-то k^2.
            //    WTF я вообще делаю
            //    Зачем эти сигмы, почему нельзя оперировать радиусом размытия, почему всегда нужно воткнуть матанализ, нормальное распределение, e^something и все подобное
            cv::Mat imgLayer = gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES + prevLayer].clone();
#else
            double sigmaPrev = INITIAL_IMG_SIGMA * pow(2.0, octave); // sigma1  - сигма картинки с нулевого слоя октавы
            cv::Mat imgLayer = gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES].clone();
#endif
            double sigmaCur  = INITIAL_IMG_SIGMA * pow(2.0, octave) * pow(k, layer);     // sigma12 - сигма до которой мы хотим дойти на текущем слое
            double sigma = sqrt(sigmaCur*sigmaCur - sigmaPrev*sigmaPrev);                // sigma2  - сигма которую надо добавить чтобы довести sigma1 до sigma12
            cv::GaussianBlur(imgLayer, imgLayer, automaticKernelSize, sigma, sigma);
            gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES + layer] = imgLayer;
        }
    }

    for (size_t octave = 0; octave < NOCTAVES; ++octave) {
        for (size_t layer = 0; layer < OCTAVE_GAUSSIAN_IMAGES; ++layer) {
            double sigmaCur = INITIAL_IMG_SIGMA * pow(2.0, octave) * pow(k, layer);
            if (DEBUG_ENABLE) cv::imwrite(DEBUG_PATH + "pyramid/o" + to_string(octave) + "_l" + to_string(layer) + "_s" + to_string(sigmaCur) + ".png", gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES + layer]);
            // Q: какие ожидания от картинок можно придумать? т.е. как дополнительно проверить что работает разумно?
            //    спойлер: подуймайте с чем должна визуально совпадать картинка из октавы? может быть с какой-то из картинок с предыдущей октавы? с какой? как их визуально сверить ведь они разного размера?
            // A: визуально проверять как раз не проблема, в средстве просмотра выставляем на всех картинках масштаб "вписать в экран" или что-нибудь подобное - и все станут одинакового визуального размера.
        }
    }

    DoGPyramid.resize(NOCTAVES * OCTAVE_DOG_IMAGES);

    // строим пирамиду разниц гауссиан слоев (Difference of Gaussian, DoG), т.к. вычитать надо из слоя слой в рамках одной и той же октавы - то есть приятный параллелизм на уровне октав
#if PARALLEL_ENABLE
    #pragma omp parallel for
#endif
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

    // нам нужны padding-картинки по краям октавы чтобы извлекать экстремумы, но в статье предлагается не s+2 а s+3: [lowe04] We must produce s + 3 images in the stack of blurred images for each octave, so that final extrema detection covers a complete octave
    // Q: почему OCTAVE_GAUSSIAN_IMAGES=(OCTAVE_NLAYERS + 3) а не например (OCTAVE_NLAYERS + 2)?
    // A: Потому что паддинг нам нужен в DoG-картинках, а не исходных - а их на одну меньше.
    //    То есть, нам нужно две дополнительных картинки в DoG, значит в пирамиде делаем +3.

    for (size_t octave = 0; octave < NOCTAVES; ++octave) {
        for (size_t layer = 0; layer < OCTAVE_DOG_IMAGES; ++layer) {
            double sigmaCur = INITIAL_IMG_SIGMA * pow(2.0, octave) * pow(k, layer);
            if (DEBUG_ENABLE) cv::imwrite(DEBUG_PATH + "pyramidDoG/o" + to_string(octave) + "_l" + to_string(layer) + "_s" + to_string(sigmaCur) + ".png", DoGPyramid[octave * OCTAVE_DOG_IMAGES + layer]);
            // Q: какие ожидания от картинок можно придумать? т.е. как дополнительно проверить что работает разумно?
            //    спойлер: подуймайте с чем должна визуально совпадать картинка из октавы DoG? может быть с какой-то из картинок с предыдущей октавы? с какой? как их визуально сверить ведь они разного размера?
            // A: Визуально - в программе для просмотра изображений найти управление масштабом и там нажать кнопку "вписать в окно" или подобную - тогда все картинки будут одинакового визуального размера.
        }
    }
}

namespace {
    float parabolaFitting(float y0, float y1, float y2) {
        rassert((y1 >= y0 && y1 >= y2) || (y1 <= y0 && y1 <= y2), 12541241241241);

        // a*0^2+b*0+c=x0
        // a*1^2+b*1+c=x1
        // a*2^2+b*2+c=x2

        // c=x0
        // a+b+x0=x1     (2)
        // 4*a+2*b+x0=x2 (3)

        // (3)-2*(2): 2*a-y0=y2-2*y1; a=(y2-2*y1+y0)/2
        // (2):       b=y1-y0-a
        float a = (y2-2.0f*y1+y0) / 2.0f;
        float b = y1 - y0 - a;
        float c = y0;
        // extremum is at -b/(2*a), but our system coordinate start (i) is at 1, so minus 1
        float x = - b / (2.0f * a) - 1.0f;
        return x;
    }
}

void phg::SIFT::findLocalExtremasAndDescribe(const std::vector<cv::Mat> &gaussianPyramid, const std::vector<cv::Mat> &DoGPyramid,
                                             std::vector<cv::KeyPoint> &keyPoints, cv::Mat &desc) {
    std::vector<std::vector<float>> pointsDesc;

    // 3.1 Local extrema detection
#if PARALLEL_ENABLE
    #pragma omp parallel // запустили каждый вычислительный поток процессора
#endif
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
                const cv::Mat DoGs[3] = {prev, cur, next};

                // теперь каждый поток обработает свой кусок картинки
#if PARALLEL_ENABLE
                #pragma omp for
#endif
                for (ptrdiff_t y = 1; y < cur.rows - 1; ++y) {
                    for (ptrdiff_t x = 1; x + 1 < cur.cols; ++x) {
                        bool is_max = true;
                        bool is_min = true;
                        float center = DoGs[1].at<float>(y, x);
                        for (int dz = -1; dz <= 1 && (is_min || is_max); ++dz) {
                        for (int dy = -1; dy <= 1 && (is_min || is_max); ++dy) {
                        for (int dx = -1; dx <= 1 && (is_min || is_max); ++dx) {
                            // TODONE проверить является ли наш центр все еще экстремум по сравнению с соседом DoGs[1+dz].at<float>(j+dy, i+dx) ? (не забудьте учесть что один из соседей - это мы сами)
                            if (dx == 0 && dy == 0 && dz == 0) continue;
                            is_min = is_min && DoGs[1+dz].at<float>(y+dy, x+dx) > center;
                            is_max = is_max && DoGs[1+dz].at<float>(y+dy, x+dx) < center;
                        }
                        }
                        }
                        bool is_extremum = (is_min || is_max);

                        if (!is_extremum)
                            continue; // очередной элемент cascade filtering, если не экстремум - сразу заканчиваем обработку этого пикселя

                        // 4 Accurate keypoint localization
                        cv::KeyPoint kp;
                        float dx = 0.0f;
                        float dy = 0.0f;
                        float dvalue = 0.0f;
                        // TODONE сделать субпиксельное уточнение (хотя бы через параболу-фиттинг независимо по оси X и оси Y, но лучше через честный ряд Тейлора, матрицу Гессе и итеративное смещение если экстремум оказался в соседнем пикселе)
#if SUBPIXEL_FITTING_ENABLE // такие тумблеры включающие/выключающие очередное улучшение алгоритма позволяют оценить какой вклад эта фича вносит в качество результата если в рамках уже готового алгоритма попробовать ее включить/выключить
                        {
                            dx = parabolaFitting(DoGs[1].at<float>(y, x - 1), DoGs[1].at<float>(y, x), DoGs[1].at<float>(y, x + 1));
                            dy = parabolaFitting(DoGs[1].at<float>(y - 1, x), DoGs[1].at<float>(y, x), DoGs[1].at<float>(y + 1, x));
                        }
#endif
                        // TODONE сделать фильтрацию слабых точек по слабому контрасту
                        float contrast = center + dvalue;
#if CONTRAST_FILTERING_ENABLE
                        if (contrast < contrast_threshold / OCTAVE_NLAYERS)
                            continue;
#endif

                        // Q: почему порог контрастности должен уменьшаться при увеличении числа слоев в октаве?
                        // A: Потому что степень размытия слоя зависит от числа слоев - k = pow(2.0, 1.0 / OCTAVE_NLAYERS)
                        //    Чем больше слоев, тем меньше различается степень размытия соседних слоев, тем меньше слои отличаются,
                        //    тем ниже нам нужен порог чтобы обнаружить что-то полезное

                        kp.pt = cv::Point2f((x + 0.5 + dx) * octave_downscale, (y + 0.5 + dy) * octave_downscale);

                        kp.response = fabs(contrast);

                        const double k = pow(2.0, 1.0 / OCTAVE_NLAYERS); // [lowe04] k = 2^{1/s} а у нас s=OCTAVE_NLAYERS
                        double sigmaCur = INITIAL_IMG_SIGMA * pow(2.0, octave) * pow(k, layer);
                        kp.size = 2.0 * sigmaCur * 5.0;

                        // 5 Orientation assignment
                        cv::Mat img = gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES + layer];
                        std::vector<float> votes;
                        float biggestVote;
                        int oriRadius = (int) (ORIENTATION_WINDOW_R * (1.0 + k * (layer - 1)));
                        if (!buildLocalOrientationHists(img, x, y, oriRadius, votes, biggestVote))
                            continue;

                        for (size_t bin = 0; bin < ORIENTATION_NHISTS; ++bin) {
                            float prevValue = votes[(bin + ORIENTATION_NHISTS - 1) % ORIENTATION_NHISTS];
                            float value = votes[bin];
                            float nextValue = votes[(bin + 1) % ORIENTATION_NHISTS];
                            if (value > prevValue && value > nextValue && votes[bin] > biggestVote * ORIENTATION_VOTES_PEAK_RATIO) {
                                // TODONE добавьте уточнение угла наклона - может помочь определенная выше функция parabolaFitting(float x0, float x1, float x2)
                                float dbin = 0;
#if SUBBIN_FITTING_ENABLE
                                dbin = parabolaFitting(prevValue, value, nextValue);
#endif
                                kp.angle = (bin + 0.5 + dbin) * (360.0 / ORIENTATION_NHISTS);
                                rassert(kp.angle >= 0.0 && kp.angle <= 360.0, 123512412412);

                                std::vector<float> descriptor;
                                double descrSampleRadius = (DESCRIPTOR_SAMPLE_WINDOW_R * (1.0 + k * (layer - 1)));
                                if (!buildDescriptor(img, kp.pt.x, kp.pt.y, descrSampleRadius, kp.angle, descriptor))
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
        for (size_t i = 0; i < pointsDesc[i].size(); ++i) {
            desc.at<float>(j, i) = pointsDesc[j][i];
        }
    }
}

bool phg::SIFT::buildLocalOrientationHists(const cv::Mat &img, size_t i, size_t j, size_t radius,
                                           std::vector<float> &votes, float &biggestVote) {
    // 5 Orientation assignment
    votes.resize(ORIENTATION_NHISTS, 0.0f);
    biggestVote = 0.0;

    if (i-1 < radius - 1 || i+1 + radius - 1 >= img.cols || j-1 < radius - 1 || j+1 + radius - 1 >= img.rows)
        return false;

    float sum[ORIENTATION_NHISTS] = {0.0f};

    for (size_t y = j - radius + 1; y < j + radius; ++y) {
        for (size_t x = i - radius + 1; x < i + radius; ++x) {
            // m(x, y)=(L(x + 1, y) − L(x − 1, y))^2 + (L(x, y + 1) − L(x, y − 1))^2
            int dx = x - i;
            int dy = y - j;
            float l_x = img.at<float>(y, x + 1) - img.at<float>(y, x - 1);
            float l_y = img.at<float>(y + 1, x) - img.at<float>(y - 1, x);
            double magnitude = l_x * l_x + l_y * l_y;

            // orientation == theta
            // atan( (L(x, y + 1) − L(x, y − 1)) / (L(x + 1, y) − L(x − 1, y)) )
            double orientation = atan2(dy, dx);
            orientation = orientation * 180.0 / M_PI;
            orientation = (orientation + 90.0);
            if (orientation <  0.0)   orientation += 360.0;
            if (orientation >= 360.0) orientation -= 360.0;
            rassert(orientation >= 0.0 && orientation < 360.0, 5361615612);
            static_assert(360 % ORIENTATION_NHISTS == 0, "Inappropriate bins number!");
            size_t bin = orientation / 360 * ORIENTATION_NHISTS;
            rassert(bin < ORIENTATION_NHISTS, 361236315613);
            sum[bin] += magnitude;
        }
    }

    for (size_t bin = 0; bin < ORIENTATION_NHISTS; ++bin) {
        votes[bin] = sum[bin];
        // TODONE может быть сгладить получившиеся гистограммы улучшит результат?
        // A: differentiability выросло на 0.01 - 0.03 примерно.
        //    Является ли это изменение статзначимым? - оставлю без ответа.
        if (HIST_SMOOTHING_FACTOR > 0) {
            int prev_bin = (bin - 1 + ORIENTATION_NHISTS) % ORIENTATION_NHISTS;
            int next_bin = (bin + 1 + ORIENTATION_NHISTS) % ORIENTATION_NHISTS;
            votes[bin] += sum[prev_bin] * HIST_SMOOTHING_FACTOR;
            votes[bin] += sum[next_bin] * HIST_SMOOTHING_FACTOR;
        }
        biggestVote = std::max(biggestVote, sum[bin]);
    }

    return true;
}

void normalize(std::vector<float> &vector) {
    float l = 0;
    for (auto x : vector) {
        l += x * x;
    }
    for (auto &x : vector) {
        x /= l;
    }
}

bool phg::SIFT::buildDescriptor(const cv::Mat &img, float px, float py, double descrSampleRadius, float angle,
                                std::vector<float> &descriptor) {
    cv::Mat relativeShiftRotation = cv::getRotationMatrix2D(cv::Point2f(0.0f, 0.0f), -angle, 1.0);

    const double smpW = 2.0 * descrSampleRadius - 1.0;

    descriptor.resize(DESCRIPTOR_SIZE * DESCRIPTOR_SIZE * DESCRIPTOR_NBINS, 0.0f);
    for (int hst_y = 0; hst_y < DESCRIPTOR_SIZE; ++hst_y) { // перебираем строку в решетке гистограмм
        for (int hst_x = 0; hst_x < DESCRIPTOR_SIZE; ++hst_x) { // перебираем колонку в решетке гистограмм

            float sum[DESCRIPTOR_NBINS] = {0.0f};

            for (int smpj = 0; smpj < DESCRIPTOR_SAMPLES_N; ++smpj) { // перебираем строчку замера для текущей гистограммы
                for (int smpi = 0; smpi < DESCRIPTOR_SAMPLES_N; ++smpi) { // перебираем столбик очередного замера для текущей гистограммы
                    for (int smpy = 0; smpy < smpW; ++smpy) { // перебираем ряд пикселей текущего замера
                        for (int smpx = 0; smpx < smpW; ++smpx) { // перебираем столбик пикселей текущего замера
                            cv::Point2f shift(
                                ((-DESCRIPTOR_SIZE/2.0 + hst_x) * DESCRIPTOR_SAMPLES_N + smpi) * smpW,
                                ((-DESCRIPTOR_SIZE/2.0 + hst_y) * DESCRIPTOR_SAMPLES_N + smpj) * smpW
                            );
                            std::vector<cv::Point2f> shiftInVector(1, shift);
                            cv::transform(shiftInVector, shiftInVector, relativeShiftRotation); // преобразуем относительный сдвиг с учетом ориентации ключевой точки
                            shift = shiftInVector[0];

                            int x = (int) (px + shift.x);
                            int y = (int) (py + shift.y);

                            if (y - 1 < 0 || y + 1 > img.rows || x - 1 < 0 || x + 1 > img.cols)
                                return false;

                            float l_x = img.at<float>(y, x + 1) - img.at<float>(y, x - 1);
                            float l_y = img.at<float>(y + 1, x) - img.at<float>(y - 1, x);
                            double magnitude = l_x * l_x + l_y * l_y;

                            double orientation = atan2(shift.y, shift.x);
                            orientation = orientation * 180.0 / M_PI;
                            orientation = (orientation + 90.0);
                            if (orientation <  0.0)   orientation += 360.0;
                            if (orientation >= 360.0) orientation -= 360.0;

                            // Q: за счет чего этот вклад будет сравниваться с этим же вкладом даже если эта картинка будет повернута? что нужно сделать с ориентацией каждого градиента из окрестности этой ключевой точки?
                            // A: Мы перед построением дескриптора посчитали черновую ориентацию-поворот точки и передали сюда в переменной angle.
                            //    И здесь повернули картинку на -angle -- независимо от поворота картинки теперь ориентация -- вверх.

                            rassert(orientation >= 0.0 && orientation < 360.0, 3515215125412);
                            static_assert(360 % DESCRIPTOR_NBINS == 0, "Inappropriate bins number!");
                            size_t bin = (orientation / 360) * DESCRIPTOR_NBINS;
                            rassert(bin < DESCRIPTOR_NBINS, 361236315613);
                            sum[bin] += magnitude;
                            // TODO хорошая идея добавить трилинейную интерполяцию как предложено в статье, или хотя бы сэмулировать ее - сгладить получившиеся гистограммы
                        }
                    }
                }
            }

            float *votes = &(descriptor[(hst_y * DESCRIPTOR_SIZE + hst_x) * DESCRIPTOR_NBINS]); // нашли где будут лежать корзины нашей гистограммы
            for (int bin = 0; bin < DESCRIPTOR_NBINS; ++bin) {
                votes[bin] = sum[bin];
            }
        }

        // TODONE нормализовать наш вектор дескриптор (подсказка: посчитать его длину и поделить каждую компоненту на эту длину)
        normalize(descriptor);
    }



    return true;
}