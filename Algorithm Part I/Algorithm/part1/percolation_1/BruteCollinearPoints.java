/* *****************************************************************************
 *  Name:              Alan Turing
 *  Coursera User ID:  123456
 *  Last modified:     1/1/2019
 **************************************************************************** */

import java.util.Comparator;

public class BruteCollinearPoints {
    public BruteCollinearPoints(Point[] points) {
        int num = points.length;
        for (int i = 0; i < num; ++i) {
            for (int j = i + 1; j < num; ++j) {
                for (int k = j + 1; k < num; ++k) {
                    for (int h = k + 1; h < num; ++h) {
                        Point p = points[i];
                        Point q = points[j];
                        Point r = points[k];
                        Point s = points[h];
                        Comparator<Point> pc = p.slopeOrder();
                        if (pc.compare(q, r) == pc.compare(q, s)) {

                        }
                    }
                }
            }
        }
    }


    public int numberofSegments() {


    }


    public LineSegment[] segments() {


    }

    public static void main(String[] args) {

    }
}
