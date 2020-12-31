/* *****************************************************************************
 *  Name:              Alan Turing
 *  Coursera User ID:  123456
 *  Last modified:     1/1/2019
 **************************************************************************** */

import edu.princeton.cs.algs4.StdRandom;

import java.util.Iterator;
import java.util.NoSuchElementException;

public class RandomizedQueue<Item> implements Iterable<Item> {
    private Item[] data;
    private int size;
    private int current;
    // private int start;

    public RandomizedQueue() {
        data = (Item[]) new Object[8];
        size = 8;
        current = 0;
        // start = 0;
    }


    public boolean isEmpty() {
        return current == 0;
    }

    public int size() {
        return current;
    }

    public void enqueue(Item i) {
        if (i != null) {
            if (current == size - 1)
                doubleSpace();
            ++current;
            data[current - 1] = i;
        }
        else throw new IllegalArgumentException("wrong!  enqeue!");
    }

    public Item dequeue() {
        if (!isEmpty()) {
            int n = StdRandom.uniform(0, current);
            Item x = data[n];
            data[n] = data[current - 1];
            current--;
            if (current < size / 4)
                shrink();
            // else check();
            return x;
        }
        else throw new NoSuchElementException("Wrong!  dequeue  !");
    }


    public Item sample() {
        if (!isEmpty()) {
            // int t = (int) System.currentTimeMillis();
            // StdRandom.setSeed(t);
            int x = StdRandom.uniform(0, current);
            return data[x];
        }
        else throw new NoSuchElementException("Wrong ! Sample !");
    }


    private void doubleSpace() {
        Item[] old = data;
        size *= 2;
        data = (Item[]) new Object[size];
        for (int i = 0; i < current; ++i)
            data[i] = old[i];
    }

    private void shrink() {
        if (size > 4) {
            Item[] old = data;
            size /= 2;
            data = (Item[]) new Object[size];
            for (int i = 0; i < current; ++i)
                data[i] = old[i];
        }
    }

    // private void check() {
    //     if (start >= size / 4 && current >= size / 2) {
    //         Item[] old = data;
    //         data = (Item[]) new Object[size];
    //         for (int i = 0; i < current; ++i)
    //             data[i] = old[i];
    //         // start = 0;
    //     }
    // }

    public Iterator<Item> iterator() {
        return new ArrayIterator();
    }

    private class ArrayIterator implements Iterator<Item> {
        private int now;
        private final Item[] array;
        // private final boolean[] used;
        // private int change;

        public ArrayIterator() {
            // change = 0;
            now = current - 1;
            //used = new boolean[now - 1];
            array = (Item[]) new Object[current];
            // used = new boolean[current];
            for (int i = 0; i < current; ++i) {
                array[i] = data[i];
                // used[i] = false;
            }
            // for (int i = 0; change < current / 4; ++i) {
            //     int x = StdRandom.uniform(0, now);
            //     if (!used[x]) {
            //         change++;
            //         used[x] = true;
            //         used[current - 1 - x] = true;
            //         array[x] = data[current - 1 - x];
            //         array[current - 1 - x] = data[x + start];
            //     }
            // }
            // for (int i = 0; i < 10; ++i) {
            //     int x = StdRandom.uniform(0, current - 1);
            //     Item tmp = array[x];
            //     array[x] = array[current - 1];
            //     array[current - 1] = tmp;
            // }
            // now = current;
        }

        public boolean hasNext() {
            return now >= 0;
        }

        public Item next() {
            if (hasNext()) {
                int n = StdRandom.uniform(now + 1);
                Item x = array[n];
                array[n] = array[now];
                now--;
                return x;
            }
            else throw new NoSuchElementException("Next!");
        }

        public void remove() {
            throw new UnsupportedOperationException("Wrong ! Remove !");
        }
    }


    public static void main(String[] args) {
        RandomizedQueue<Integer> rq = new RandomizedQueue<Integer>();
        System.out.println("Test is going! Let's Go!---------------------------");

        rq.enqueue(18);
        rq.enqueue(17);
        rq.enqueue(16);
        rq.enqueue(15);
        rq.enqueue(14);
        rq.enqueue(13);
        rq.enqueue(12);
        rq.enqueue(11);
        rq.enqueue(10);
        rq.enqueue(9);
        rq.enqueue(8);
        rq.enqueue(7);
        rq.enqueue(6);
        rq.enqueue(5);
        rq.enqueue(4);
        rq.enqueue(3);
        rq.enqueue(2);
        rq.enqueue(1);


        //
        // System.out.println("En queue");

        //
        System.out.println("Iterator : --------");
        for (int i : rq)
            System.out.print(i + " ");
        System.out.println();


        System.out.println("Iterator : --------");
        for (int i : rq)
            System.out.print(i + " ");
        System.out.println();


        System.out.println(rq.dequeue());
        System.out.println(rq.dequeue());
        System.out.println(rq.dequeue());
        System.out.println(rq.dequeue());
        System.out.println(rq.dequeue());

        System.out.println("Iterator : --------");
        for (int i : rq) {
            for (int j : rq) {
                System.out.print(i + "-" + j + " ");
            }
            System.out.println();
            // System.out.println();
        }
        System.out.println();

        //
        // rq.enqueue(2);
        //
        // System.out.println("Iterator : --------");
        // for (int i : rq)
        //     System.out.print(i + " ");
        // System.out.println();
        //
        // rq.enqueue(3);
        //
        // System.out.println("Iterator : --------");
        // for (int i : rq)
        //     System.out.print(i + " ");
        // System.out.println();
        //
        // rq.enqueue(4);
        //
        // System.out.println("Iterator : --------");
        // for (int i : rq)
        //     System.out.print(i + " ");
        // System.out.println();
        //
        //
        // rq.enqueue(5);
        //
        // System.out.println(rq.size());
        // System.out.println("Iterator : --------");
        // for (int i : rq)
        //     System.out.print(i + " ");
        // System.out.println();
        //
        //
        // System.out.println("Sample :");
        // for (int i = 0; i < 20; ++i)
        //     System.out.print(rq.sample() + " ");
        // System.out.println();
        //
        //
        // System.out.println("De queue :");
        // System.out.println("          " + rq.dequeue());
        //
        // System.out.println("Sample :");
        // for (int i = 0; i < 20; ++i)
        //     System.out.print(rq.sample() + " ");
        // System.out.println();
        //
        //
        // System.out.println("          " + rq.dequeue());
        // System.out.println(rq.size());
        //
        //
        // for (int i : rq)
        //     System.out.print(i + " ");
        // System.out.println();

    }
}

