/* *****************************************************************************
 *  Name:              Alan Turing
 *  Coursera User ID:  123456
 *  Last modified:     1/1/2019
 **************************************************************************** */

public class UnionFind {
    public static void main(String[] args) {

    }

    private int[] id;

    public UnionFind(int N) {
        id = new int[N];
        for (int i = 0; i < N; ++i) id[i] = i;
    }

    private int root(int i) {
        while (i != id[i]) i = id[i];
        return i;
        // return id[i];
    }

    public boolean connected(int p, int q) {
        return root(p) == root(q);
    }

    public void union(int p, int q) {
        int i = root(p);
        int j = root(q);
        id[i] = j;
    }
}
