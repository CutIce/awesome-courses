/* *****************************************************************************
 *  Name:              Alan Turing
 *  Coursera User ID:  123456
 *  Last modified:     1/1/2019
 **************************************************************************** */

import edu.princeton.cs.algs4.StdIn;
import edu.princeton.cs.algs4.StdOut;

public class Permutation {
    public static void main(String[] args) {
        int k = Integer.parseInt(args[0]);
        // In in = new In(args[1]);
        RandomizedQueue<String> rq = new RandomizedQueue<String>();

        String s;
        while (!StdIn.isEmpty()) {
            s = StdIn.readString();
            rq.enqueue(s);
        }
        int num = 0;
        for (String ss : rq) {
            if (num < k) {
                StdOut.println(ss);
                num++;
            }
            else break;
        }

    }
}
