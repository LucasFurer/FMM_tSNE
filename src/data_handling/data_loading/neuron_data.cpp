#include <FMM_tSNE/data_handling/data_loading/neuron_data.h>

#include <FMM_tSNE/data_handling/data_entries/sparse_matrix.h>
#include <algorithm>
#include <cstdint>
#include <highfive/H5File.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace NeuronData
{
namespace detail
{
// 10x v1/v2 files keep the matrix in a group named after the genome ("mm10", "GRCh38", ...);
// v3 files use "matrix". Either way the layout inside is identical.
std::string find_10x_group(const HighFive::File& file)
{
    const std::vector<std::string> names = file.listObjectNames();
    if (std::find(names.begin(), names.end(), "matrix") != names.end()) return "matrix";
    for (const std::string& name : names)
    {
        if (file.getObjectType(name) != HighFive::ObjectType::Group) continue;
        if (file.getGroup(name).exist("indptr")) return name;
    }
    throw std::runtime_error("load_10x_h5: no group with an 'indptr' dataset found");
}
} // namespace detail

// Loads a 10x Genomics HDF5 count matrix as a cells x genes CSR matrix, never densifying.
//
// The file stores genes x cells in CSC order, i.e. all entries of one cell are contiguous.
// Read as CSR that is exactly "one row per cell", so data/indices/indptr map 1:1 onto
// values/col_idx/row_ptr and no transposition is needed. The data is read straight from
// HDF5 into the final arrays (HDF5 converts int32 -> float and int64 -> uint32 on the fly).
//
// first_cell / n_cells select a contiguous range of cells (default: all of them).
// group = "" auto-detects the group ("mm10" / "matrix" / ...).
SparseMatrix load_10x_h5(const std::string& path, size_t first_cell, size_t n_cells, const std::string& group)
{
    HighFive::File file(path, HighFive::File::ReadOnly);
    HighFive::Group g = file.getGroup(group.empty() ? detail::find_10x_group(file) : group);

    std::vector<int64_t> shape;
    g.getDataSet("shape").read(shape);
    if (shape.size() != 2) throw std::runtime_error("load_10x_h5: 'shape' must have 2 entries");
    const auto n_genes = static_cast<size_t>(shape[0]);
    const auto total_cells = static_cast<size_t>(shape[1]);

    if (first_cell > total_cells) throw std::out_of_range("load_10x_h5: first_cell beyond the last cell");
    n_cells = std::min(n_cells, total_cells - first_cell);

    // indptr slice: n_cells + 1 entries, relative to the start of this cell range
    std::vector<int64_t> indptr(n_cells + 1);
    g.getDataSet("indptr")
        .select(std::vector<size_t>{first_cell}, std::vector<size_t>{n_cells + 1})
        .read_raw(indptr.data());

    const auto base = static_cast<size_t>(indptr.front());
    const auto nnz = static_cast<size_t>(indptr.back()) - base;

    SparseMatrix out;
    out.n_rows = n_cells;
    out.n_cols = n_genes;
    out.row_ptr.resize(n_cells + 1);
    for (size_t i = 0; i <= n_cells; ++i)
        out.row_ptr[i] = static_cast<uint64_t>(indptr[i]) - base;
    std::vector<int64_t>().swap(indptr);

    out.col_idx.resize(nnz);
    out.values.resize(nnz);

    // Read in chunks (~67M entries) so a single H5Dread never has to be gigantic.
    constexpr size_t kChunk = size_t{1} << 26;
    const HighFive::DataSet ds_indices = g.getDataSet("indices");
    const HighFive::DataSet ds_data = g.getDataSet("data");
    for (size_t off = 0; off < nnz; off += kChunk)
    {
        const size_t cnt = std::min(kChunk, nnz - off);
        const std::vector<size_t> o{base + off};
        const std::vector<size_t> c{cnt};
        ds_indices.select(o, c).read_raw(out.col_idx.data() + off);
        ds_data.select(o, c).read_raw(out.values.data() + off);
    }

    // The VP-tree's sparse distance needs sorted rows. 10x files are normally sorted already.
    if (!out.rows_sorted()) out.sort_rows();

    return out;
}
} // namespace NeuronData
