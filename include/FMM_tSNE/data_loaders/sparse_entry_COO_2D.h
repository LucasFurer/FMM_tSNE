#pragma once

struct SparseEntryCOO2D
{
    int col;
    int row;
    double val;

    SparseEntryCOO2D()
    {
        col = 0;
        row = 0;
        val = 0.0;
    }

    SparseEntryCOO2D(int initCol, int initRow, double initVal)
    {
        col = initCol;
        row = initRow;
        val = initVal;
    }
};
