/* *****************************************************************************
 *  Name:              CutIce
 *  Coursera User ID:  CutIce
 *  Last modified:     8/9/2020
 **************************************************************************** */

import edu.princeton.cs.algs4.StdRandom;
import edu.princeton.cs.algs4.StdStats;

public class PercolationStats {
    private double meanval;
    private double stddevval;
    private double cfdsLowVal;
    private double cfdsHighVal;

    public PercolationStats(int n, int t) {
        // long st = System.currentTimeMillis();
        if (validate(n, t)) {
            double con = 1.96;
            double[] result = new double[t];
            meanval = 0.0;
            stddevval = 0.0;
            cfdsLowVal = 0.0;
            cfdsHighVal = 0.0;
            double allnum = n * n;
            for (int k = 0; k < t; ++k) {
                Percolation simulation = new Percolation(n);
                while (!simulation.percolates()) {
                    int x = StdRandom.uniform(1, n + 1);
                    int y = StdRandom.uniform(1, n + 1);
                    //
                    simulation.open(x, y);
                }
                int countnum = simulation.numberOfOpenSites();
                result[k] = (double) countnum / allnum;
            }
            double sqrttimes = Math.sqrt(t);
            meanval = StdStats.mean(result);
            stddevval = StdStats.stddev(result);
            cfdsLowVal = meanval - con * stddevval / sqrttimes;
            cfdsHighVal = meanval + con * stddevval / sqrttimes;
            // long et = System.currentTimeMillis();
            // System.out.print("Time: ");
            // System.out.println(et - st);
        }
        else throw new IllegalArgumentException("Wrong!  Status!");
    }


    public double mean() {
        return meanval;
    }


    public double stddev() {
        return stddevval;
    }

    public double confidenceLo() {
        return cfdsLowVal;
    }

    public double confidenceHi() {
        return cfdsHighVal;
    }

    private boolean validate(int i, int j) {
        return i > 0 && j > 0;
    }


    public static void main(String[] args) {
        // int x = 0;
        // for (int i = 0; i < args.length; ++i) {
        //     x = 10 * x + Integer.parseInt(args[i]);
        // }
        // System.out.println(x);
        // PercolationStats p = new PercolationStats(400, x);
        // System.out.println("mean: " + p.mean());
        // System.out.println("stddev: " + p.stddev());
        // System.out.println("Low: " + p.confidenceLo());
        // System.out.println("High: " + p.confidenceHi());
        int x1 = Integer.parseInt(args[0]);
        int x2 = Integer.parseInt(args[1]);
        PercolationStats p = new PercolationStats(x1, x2);
        System.out.println("mean                   = " + p.mean());
        System.out.println("stddev                 = " + p.stddev());
        System.out.println("95% confidence interval=[" + p.cfdsLowVal + "," + p.cfdsHighVal + "]");
    }
}
