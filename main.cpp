#include <vector>
#include <random>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include "misc/logging/logging.h"

DEFINE_int32(grid_size, 3, "size of the bezied control grid");
DEFINE_double(sigma, 25.0, "control points variance");
DEFINE_bool(verbose, false, "Be verbose");

const double kBernsteinPolynomialMaxDegree = 10;

inline unsigned int n_choose_k(unsigned int n, unsigned int k) {
    unsigned int nchoosek = 1;
    for (unsigned int i = n - k + 1; i <= n; ++i) {
        nchoosek *= i;
    }
    for (unsigned int i = 1; i <= k; ++i) {
        nchoosek /= i;
    }
    return nchoosek;
}

double BernsteinPolynomialEvaluation(unsigned int degree, unsigned int index, double x) {
    // check that the degree is not too high otherwise nchoosek
    // function would take too high value
    if (degree > kBernsteinPolynomialMaxDegree) {
        LOG(WARNING) << "Bernstein polynomial: " << degree << ", " << index
                  << " degree too high, return 0 vector";
    }
    // check that the berstein polynom is defined
    if (index > degree) {
        LOG(WARNING) << "Bernstein polynomial: " << degree << ", " << index
                  << " not defined, return 0 vector";
    }

    double a = (index == 0) ? 1.0 : pow(x, index);
    double b = (degree - index == 0) ? 1.0 : pow(1.0 - x, degree - index);
    return n_choose_k(degree, index) * a * b;
}

class RandomBezierGrid {
public:
    RandomBezierGrid(int n1, int n2, int width, int height, double sigma = 0.05) {
        // init grid vector
        control_grid_ = std::vector<std::vector<cv::Vec2d>>(n1, std::vector<cv::Vec2d>(n2));

        // init dim
        n1_ = n1;
        n2_ = n2;

        // random generator
        std::default_random_engine generator;
        std::normal_distribution<double> distribution(0.0, sigma);

        // fill control points
        for (int i = 0; i < n1; ++i) {
            const double x = static_cast<double>(width) * static_cast<double>(i) / static_cast<double>(n1_ - 1);
            for (int j = 0; j < n2; ++j) {
                const double y = static_cast<double>(height) * static_cast<double>(j) / static_cast<double>(n2_ - 1);
                control_grid_[i][j] = cv::Vec2d(x + distribution(generator), y + distribution(generator));
            }
        }
    }

    cv::Vec2d Evaluate(cv::Vec2d uv, int width, int height) {
        const double u = uv(0) / static_cast<double>(width - 1);
        const double v = uv(1) / static_cast<double>(height - 1);
        cv::Vec2d result(0.0, 0.0);
        for (int i = 0; i < n1_; ++i) {
            for (int j = 0; j < n2_; ++j) {
                const double bernstein_1 = BernsteinPolynomialEvaluation(n1_ - 1, i, u);
                const double bernstein_2 = BernsteinPolynomialEvaluation(n2_ - 1, j, v);
                const cv::Vec2d value = bernstein_1 * bernstein_2 * control_grid_[i][j];
                result += value;
            }
        }

        return result;
    }


private:
    std::vector<std::vector<cv::Vec2d>> control_grid_;
    int n1_;
    int n2_;
};


int main(int argc, char ** argv) {
    google::SetUsageMessage("[OPTIONS] input_image output_image");
    google::SetVersionString("1.0");
    google::InitGoogleLogging("GridDistortionAugmentation");
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::SetStderrLogging(FLAGS_verbose ? google::INFO : google::WARNING);

    // Check arguments
    if (argc != 3) {
        LOG(FATAL) << "Usage " << argv[0] << " input_image output_image";
    }

    // Load input image
    cv::Mat image = cv::imread(std::string(argv[1]));
    if (image.empty()) {
        LOG(FATAL) << "Could not load image: " << argv[1];
    }

    // Init random bezier class
    RandomBezierGrid bezier(FLAGS_grid_size, FLAGS_grid_size, image.size().width, image.size().height, FLAGS_sigma);

    // Fill mapping
    cv::Mat x_map = cv::Mat(image.size(), CV_32F);
    cv::Mat y_map = cv::Mat(image.size(), CV_32F);
    for (int i = 0; i < image.size().width; ++i) {
        for (int j = 0; j < image.size().height; ++j) {
            const cv::Vec2d uv(i, j);
            const cv::Vec2d distorted = bezier.Evaluate(uv, image.size().width, image.size().height);
            x_map.at<float>(j, i) = distorted(0);
            y_map.at<float>(j, i) = distorted(1);
        }
    }

    // Apply mapping
    cv::Mat result;
    cv::remap(image, result, x_map, y_map, cv::INTER_AREA);

    // Save image
    cv::imwrite(std::string(argv[2]), result);

    return EXIT_SUCCESS;
}
