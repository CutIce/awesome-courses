/*
 *****************************************************************************
 *  Name:              CutIce
 *  Coursera User ID:  CutIce
 *  Last modified:     9/11/2020
 ****************************************************************************


import edu.princeton.cs.algs4.In;
import edu.princeton.cs.algs4.StdDraw;
import edu.princeton.cs.algs4.StdOut;

public class FastCollinearPoints {
    private int number;

    public static void main(String[] args) {
        // read the n points from a file
        In in = new In(args[0]);
        int n = in.readInt();
        Point[] points = new Point[n];
        for (int i = 0; i < n; i++) {
            int x = in.readInt();
            int y = in.readInt();
            points[i] = new Point(x, y);
        }

        // draw the points
        StdDraw.enableDoubleBuffering();
        StdDraw.setXscale(0, 32768);
        StdDraw.setYscale(0, 32768);
        for (Point p : points) {
            p.draw();
        }
        StdDraw.show();

        // print and draw the line segments
        FastCollinearPoints collinear = new FastCollinearPoints(points);
        for (LineSegment segment : collinear.segments()) {
            StdOut.println(segment);
            segment.draw();
        }
        StdDraw.show();
    }

    public FastCollinearPoints(Point[] Points) {
        if (Points == null) throw new IllegalArgumentException("Null inout ub FastCollinearPoints");
        number = 0;
        int num = Points.length;
        for (int i = 0; i < num; ++i) {

        }
    }

    public int numberOfSegments() {
        return number;
    }

    public LineSegment[] segments() {

    }
}
*/

import java.util.ArrayList;
import java.util.Arrays;

public class FastCollinearPoints {
    private final Point[] points;
    private final LineSegment[] cached;

    public FastCollinearPoints(Point[] points) {
        if (points == null) {
            throw new IllegalArgumentException();
        }
        // Points array passsed to the constructor can be changed by some other parts of the code while construction is still in progress.
        this.points = Arrays.copyOf(points, points.length);
        for (Point point : this.points) {
            if (point == null) {
                throw new IllegalArgumentException();
            }
        }
        Arrays.sort(this.points);
        for (int i = 0; i < this.points.length; i++) {
            if (i > 0 && Double.compare(this.points[i].slopeTo(this.points[i - 1]),
                                        Double.NEGATIVE_INFINITY) == 0) {
                throw new IllegalArgumentException();
            }
        }
        // Stores a reference to an externally mutable object in the instance variable 'points', exposing the internal representation of the class.
        // Instead, create a defensive copy of the object referenced by the parameter variable 'points' and store that copy in the instance variable 'points'.
        cached = cache();
    }

    public int numberOfSegments() {
        return cached.length;
    }

    public LineSegment[] segments() {
        // check that data type is immutable by testing whether each method
        // returns the same value, regardless of any intervening operations
        return Arrays.copyOf(cached, cached.length);
    }

    private LineSegment[] cache() {
        ArrayList<LineSegment> list = new ArrayList<>();
        Arrays.sort(points);
        for (Point p : points) {
            Point[] pp = Arrays.copyOf(points, points.length);
            if (pp.length < 4) {
                continue;
            }
            Arrays.sort(pp, p.slopeOrder());
            int begin = 1;
            int end = 1;
            double last = p.slopeTo(pp[end]);
            for (int j = 2; j < pp.length; j++) {
                double slope = p.slopeTo(pp[j]);
                if (Double.compare(last, slope) != 0) {
                    if (end - begin >= 2) {
                        // end - begin + 1 有 3 个以上了，加上 p（即pp[0]） 就至少有 4 个了
                        if (p.compareTo(pp[begin]) < 0) {
                            // 去掉子线段
                            /*
                            Let's say you have 5 points in their natural order a, b, c, d, e.
                            When you have b as the anchor and sort the remaining points by slopeOrder, you want points with the same slope to appear in their natural order (i.e. ... a, c, d, e, ...).
                            To avoid adding the subsegment (b, c, d, e), whenever you start seeing a new slope (i.e. at a), you just need to check if b is less than a in terms of natural order
                            - if it is not, it means b is not the first point in that segment, then you know you don't need to add it.
                             */
                            list.add(new LineSegment(p, pp[end]));
                        }
                    }
                    last = slope;
                    begin = j;
                }
                end = j;
            }
            if (end - begin >= 2) {
                if (p.compareTo(pp[begin]) < 0) {
                    list.add(new LineSegment(p, pp[end]));
                }
            }
        }
        LineSegment[] segments = new LineSegment[list.size()];
        return list.toArray(segments);
    }
}
