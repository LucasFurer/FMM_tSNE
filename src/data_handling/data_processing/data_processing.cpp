#include <FMM_tSNE/data_handling/data_processing/data_processing.h>

#include <Eigen/Dense>
#include <stdexcept>
#include <vector>

namespace DataProcessing
{
std::vector<std::vector<float>> PCA(const std::vector<std::vector<float>>& input, int n_components = 50)
{
    if (input.empty() || input[0].empty()) throw std::invalid_argument("PCA: empty input");

    const Eigen::Index n = static_cast<Eigen::Index>(input.size());
    const Eigen::Index d = static_cast<Eigen::Index>(input[0].size());
    const Eigen::Index k = std::min<Eigen::Index>(n_components, std::min(n, d));

    // 1. Copy into an Eigen matrix (float to keep memory ~220 MB instead of ~440 MB)
    Eigen::MatrixXf X(n, d);
    for (Eigen::Index i = 0; i < n; ++i)
    {
        if (static_cast<Eigen::Index>(input[i].size()) != d)
            throw std::invalid_argument("PCA: rows have inconsistent sizes");
        X.row(i) = Eigen::Map<const Eigen::RowVectorXf>(input[i].data(), d);
    }

    // 2. Center the data
    Eigen::RowVectorXf mean = X.colwise().mean();
    X.rowwise() -= mean;

    // 3. SVD (only V is needed, so skip U to save memory and time)
    Eigen::BDCSVD<Eigen::MatrixXf> svd(X, Eigen::ComputeThinV);

    // 4. Principal axes = first k right singular vectors (d x k)
    Eigen::MatrixXf components = svd.matrixV().leftCols(k);

    // 5. Project: (n x d) * (d x k) = (n x k)
    Eigen::MatrixXf projected = X * components;

    // 6. Copy back to vector<vector<float>>
    std::vector<std::vector<float>> output(n, std::vector<float>(k));
    for (Eigen::Index i = 0; i < n; ++i)
        Eigen::Map<Eigen::RowVectorXf>(output[i].data(), k) = projected.row(i);

    return output;
}
} // namespace DataProcessing
