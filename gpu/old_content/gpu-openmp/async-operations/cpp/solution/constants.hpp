#pragma omp declare target
constexpr int width = 8192;
constexpr int height = 8192;
constexpr int max_iters = 100;
constexpr double xmin = -1.7;
constexpr double xmax = .5;
constexpr double ymin = -1.2;
constexpr double ymax = 1.2;
constexpr double dx = (xmax - xmin) / width;
constexpr double dy =( ymax - ymin) / height;
#pragma omp end declare target
