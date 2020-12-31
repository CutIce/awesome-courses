/******************************************************************************
 *  Compilation:  javac Point.java
 *  Execution:    java Point
 *  Dependencies: none
 *
 *  An immutable data type for points in the plane.
 *  For use on Coursera, Algorithms Part I programming assignment.
 *
 ******************************************************************************/

import edu.princeton.cs.algs4.StdDraw;
import edu.princeton.cs.algs4.StdRandom;

import java.util.Comparator;

public class Point implements Comparable<Point> {

    private final int x;     // x-coordinate of this point
    private final int y;     // y-coordinate of this point

    public Point(int x, int y) {
        /* DO NOT MODIFY */
        this.x = x;
        this.y = y;
    }

    public void draw() {
        /* DO NOT MODIFY */
        StdDraw.point(x, y);
    }

    public void drawTo(Point that) {
        /* DO NOT MODIFY */
        StdDraw.line(this.x, this.y, that.x, that.y);
    }


    public double slopeTo(Point that) {
        if (that.x - x == 0 && that.y - y == 0) return Double.NEGATIVE_INFINITY;
        if (that.x - x == 0) return Double.POSITIVE_INFINITY;
        if (that.y - y == 0) return 0.0;
        return (double) (that.y - y) / (that.x - x);
    }


    public int compareTo(Point that) {
        if (y < that.y) return -1;
        if (y > that.y) return 1;
        return Integer.compare(x, that.x);
    }


    public String toString() {

        return "(" + x + ", " + y + ")";
    }


    public Comparator<Point> slopeOrder() {
        return new SlopeOrder();
    }

    private class SlopeOrder implements Comparator<Point> {
        public int compare(Point a, Point b) {
            double v1 = slopeTo(a);
            double v2 = slopeTo(b);
            if (v1 == v2) return 0;
            if (v1 < v2) return -1;
            return +1;
        }
    }

    public static void main(String[] args) {
        for (int i = 0; i < 100; ++i) {
            System.out.print(StdRandom.uniform(100, 200) + ", ");
        }

        System.out.println();

        for (int i = 0; i < 20; ++i) {
            System.out.print(StdRandom.uniform(-50, 150) + ", ");
        }

        System.out.println();


        for (int i = 0; i < 20; ++i) {
            System.out.print(StdRandom.uniform(0, 10) + ", ");
        }

        System.out.println();
    }
}
