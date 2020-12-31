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
        if (that.x == x && that.y == y) return Double.NEGATIVE_INFINITY;
        if (that.x == x) return Double.POSITIVE_INFINITY;
        if (that.y - y == 0) return +0;
        return (double) (that.y - y) / (that.x - x);
    }


    public int compareTo(Point that) {
        if (this.y < that.y) return -1;
        if (this.y > that.y) return 1;
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
            return Double.compare(v1, v2);
        }
    }


    public static void main(String[] args) {

    }
}
