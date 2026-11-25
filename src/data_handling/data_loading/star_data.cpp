#include <FMM_tSNE/data_handling/data_loading/star_data.h>

#include <fstream>
#include <limits>
#include <print>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace StarData
{
using Row = std::vector<float>;
using Table = std::vector<Row>;

void add_invalid_cols_from_line(const std::string& line, std::unordered_set<size_t>& invalid_cols)
{
    size_t i_col{};
    std::string field;

    auto process_field = [&invalid_cols, &i_col](std::string& field) -> void
    {
        if (field.empty())
        {
            invalid_cols.insert(i_col);
        }
        else if (!invalid_cols.contains(i_col))
        {
            try
            {
                float converted_field = std::stof(field);
            }
            catch (...)
            {
                invalid_cols.insert(i_col);
            }
        }
        field.clear();
    };

    for (size_t i{}; i < line.size(); ++i)
    {
        const char c = line[i];

        if (c == ',')
        {
            process_field(field);
            ++i_col;
        }
        else
        {
            field += c;
        }
    }
    process_field(field);
}

Row parse_line(const std::string& line, const std::unordered_set<size_t>& invalid_cols)
{
    size_t i_col{};
    Row fields;
    std::string field;

    auto process_field = [&invalid_cols, &i_col, &fields](std::string& field) -> void
    {
        if (!invalid_cols.contains(i_col))
        {
            float converted_field = std::stof(field);
            fields.push_back(converted_field);
        }
        field.clear();
    };

    for (size_t i{}; i < line.size(); ++i)
    {
        const char c = line[i];

        if (c == ',')
        {
            process_field(field);
            ++i_col;
        }
        else
        {
            field += c;
        }
    }
    process_field(field);

    return fields;
}

std::vector<std::vector<float>>& normalize(std::vector<std::vector<float>>& data)
{
    std::vector<float> col_mins(data[0].size(), std::numeric_limits<float>::max());
    std::vector<float> col_maxs(data[0].size(), std::numeric_limits<float>::lowest());

    for (const Row& row : data)
    {
        for (size_t c{}; c < row.size(); ++c)
        {
            col_mins[c] = std::min(col_mins[c], row[c]);
            col_maxs[c] = std::max(col_maxs[c], row[c]);
        }
    }

    for (Row& row : data)
    {
        for (size_t c{}; c < row.size(); ++c)
        {
            if ((col_maxs[c] - col_mins[c]) != 0.0f) row[c] = (row[c] - col_mins[c]) / (col_maxs[c] - col_mins[c]);
        }
    }

    return data;
}

std::vector<std::vector<float>> load_star_data(const std::vector<std::string>& data)
{
    std::unordered_set<size_t> invalid_cols;
    Table table;

    for (const std::string& line : data)
    {
        add_invalid_cols_from_line(line, invalid_cols);
    }

    for (const std::string& line : data)
    {
        table.push_back(parse_line(line, invalid_cols));
    }

    return normalize(table);
}

std::vector<std::vector<float>> load_full_star_data(size_t row_limit, const std::string& path)
{
    std::vector<std::string> files{
        "TgasSource_000-000-000.csv",
        "TgasSource_000-000-001.csv",
        "TgasSource_000-000-002.csv",
        "TgasSource_000-000-003.csv",
        "TgasSource_000-000-004.csv",
        "TgasSource_000-000-005.csv",
        "TgasSource_000-000-006.csv",
        "TgasSource_000-000-007.csv",
        "TgasSource_000-000-008.csv",
        "TgasSource_000-000-009.csv",
        "TgasSource_000-000-010.csv",
        "TgasSource_000-000-011.csv",
        "TgasSource_000-000-012.csv",
        "TgasSource_000-000-013.csv",
        "TgasSource_000-000-014.csv",
        "TgasSource_000-000-015.csv",
    };

    std::vector<std::string> data;

    for (size_t f{}; f < files.size(); ++f)
    {
        std::ifstream in_file(path + files[f]);
        if (!in_file)
        {
            throw std::runtime_error("Could not open file: " + path);
        }

        std::string line;

        std::getline(in_file, line); // get rid of the first row
        while (std::getline(in_file, line))
        {
            data.push_back(line);
        }
    }

    data.resize(std::min(row_limit, data.size()));
    return load_star_data(data);
}
} // namespace StarData
