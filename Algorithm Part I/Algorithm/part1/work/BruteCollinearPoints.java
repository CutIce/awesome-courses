import edu.princeton.cs.algs4.In;
import edu.princeton.cs.algs4.StdDraw;
import edu.princeton.cs.algs4.StdOut;

import java.util.ArrayList;
import java.util.Arrays;

public class BruteCollinearPoints {
    private final LineSegment[] store;
    private final Point[] points;

    public BruteCollinearPoints(Point[] points) {
        if (points == null) {
            throw new IllegalArgumentException();
        }
        this.points = Arrays.copyOf(points, points.length);
        for (Point point : this.points) {
            if (point == null) {
                throw new IllegalArgumentException();
            }
        }
        Arrays.sort(this.points);

        for (int i = 0; i < this.points.length; ++i) {
            if (i > 0 && Double.compare(this.points[i].slopeTo(this.points[i - 1]),
                                        Double.NEGATIVE_INFINITY) == 0) {
                throw new IllegalArgumentException();
            }
        }

        ArrayList<LineSegment> list = new ArrayList<>();
        int num = points.length;


        for (int i = 0; i < num; ++i) {
            Point p = points[i];
            for (int j = i + 1; j < num; ++j) {
                Point q = points[j];
                for (int k = j + 1; k < num; ++k) {
                    Point r = points[k];
                    for (int h = k + 1; h < num; ++h) {
                        Point s = points[h];

                        if (p.slopeTo(q) == p.slopeTo(r) && p.slopeTo(q) == p.slopeTo(s)) {
                            list.add(new LineSegment(p, s));
                        }
                    }
                }
            }
        }
        store = new LineSegment[list.size()];
        list.toArray(store);
    }


    public int numberOfSegments() {
        return store.length;
    }


    public LineSegment[] segments() {
        return Arrays.copyOf(store, store.length);
    }

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

}
