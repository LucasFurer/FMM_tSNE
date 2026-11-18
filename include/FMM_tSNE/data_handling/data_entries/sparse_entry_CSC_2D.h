#pragma once

struct SparseEntryCSC2D
{
    int row;
    double val;

    SparseEntryCSC2D()
    {
        row = 0;
        val = 0.0;
    }

    SparseEntryCSC2D(int initRow, double initVal)
    {
        row = initRow;
        val = initVal;
    }
};
