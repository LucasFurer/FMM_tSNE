#pragma once

#include <FMM_tSNE/data_handling/data_entries/sparse_matrix.h>
#include <highfive/H5File.hpp>
#include <limits>
#include <string>

namespace NeuronData
{
SparseMatrix load_10x_h5(const std::string& path = "TSNE_data_input/neuron/1M_neurons_filtered_gene_bc_matrices_h5.h5",
                         size_t first_cell = 0,
                         size_t n_cells = std::numeric_limits<size_t>::max(),
                         const std::string& group = "");
} // namespace NeuronData
