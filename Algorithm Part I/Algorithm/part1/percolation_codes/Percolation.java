/* *****************************************************************************
 *  Name:              CutIce
 *  Coursera User ID:  CutIce
 *  Last modified:     8/9/2020
 **************************************************************************** */

import edu.princeton.cs.algs4.WeightedQuickUnionUF;

public class Percolation {
    private final int size;
    private final int source;
    private final int sink;
    private int count;
    private boolean[][] data;
    /*
        in this array we use 0 -> full, 1 -> open
        and we use the 0-th line represent the water source
        use the n+1-th line represent the bottom source
    */
    private final WeightedQuickUnionUF uptodown;
    private final WeightedQuickUnionUF back;

    private final int[][] dx = { { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 } };
    // private final int[] dy = { 0, 0, 1, -1 };

    public Percolation(int n) {
        this.size = n;
        if (n <= 0) {
            throw new IllegalArgumentException("Wrong!  Percolation!");
        }
        source = n * n;
        count = 0;
        sink = n * n + 1;
        data = new boolean[n + 1][n + 1];
        for (int i = 0; i < n + 1; ++i)
            for (int j = 0; j < n + 1; ++j) data[i][j] = false;

        uptodown = new WeightedQuickUnionUF(sink + 2);
        back = new WeightedQuickUnionUF(sink + 2);

    }

    public static void main(String[] args) {
        // no one
    }

    public void open(int i, int j) {
        if (validate(i, j)) {
            if (data[i - 1][j - 1]) return;
            count++;
            data[i - 1][j - 1] = true;
            // check the (i,j) site's attribute
            if (i == 1) {
                back.union((i - 1) * size + j - 1, source);
                uptodown.union((i - 1) * size + j - 1, source);
            }
            if (i == size) {
                uptodown.union((i - 1) * size + j - 1, sink);
            }
            // union adjecent sites near the (i,j)
            for (int k = 0; k < dx.length; ++k) {
                int xx = i + dx[k][0], yy = j + dx[k][1];
                if (validate(xx, yy)) {
                    if (isOpen(xx, yy)) {
                        uptodown.union((i - 1) * size + j - 1, (xx - 1) * size + yy - 1);
                        back.union((i - 1) * size + j - 1, (xx - 1) * size + yy - 1);
                    }
                }
            }

        }
        else throw new IllegalArgumentException("Wrong!  Open!");
    }

    public boolean isOpen(int i, int j) {
        if (validate(i, j))
            return data[i - 1][j - 1];
        else throw new IllegalArgumentException("Wrong!  isOpen()!");
    }

    public boolean isFull(int i, int j) {
        if (validate(i, j)) {
            if (isOpen(i, j) && (back.find((i - 1) * size + j - 1) == back.find(source)))
                return true;
            return false;
        }
        else throw new IllegalArgumentException("Wrong!  isFull");
    }


    public boolean percolates() {
        return uptodown.find(source) == uptodown.find(sink);
    }

    private boolean validate(int i, int j) {
        if (i < 1 || i > size || j < 1 || j > size)
            return false;
        return true;
    }

    public int numberOfOpenSites() {
        return count;
    }

}
