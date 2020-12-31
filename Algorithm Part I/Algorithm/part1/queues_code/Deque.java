/* *****************************************************************************
 *  Name:              CutIce
 *  Coursera User ID:  CutIce
 *  Last modified:     8/10/2020
 **************************************************************************** */

import java.util.Iterator;
import java.util.NoSuchElementException;

public class Deque<Item> implements Iterable<Item> {
    private int size;

    private class Node {
        private final Item data;
        private Node prev;
        private Node next;

        public Node(Item d, Node p, Node n) {
            data = d;
            prev = p;
            next = n;
        }
    }

    private Node head;
    private Node tail;

    public Deque() {
        size = 0;
        head = new Node(null, null, null);
        tail = new Node(null, null, null);
        head.next = tail;
        tail.prev = head;
    }

    public boolean isEmpty() {
        return size == 0;
    }

    public int size() {
        return size;
    }

    public void addFirst(Item item) {
        if (item != null) {
            size++;
            Node newone = new Node(item, head, head.next);
            head.next.prev = newone;
            head.next = newone;
        }
        else throw new IllegalArgumentException("Wrong !  addFirst  !");
    }

    public void addLast(Item item) {
        if (item != null) {
            size++;
            Node newone = new Node(item, tail.prev, tail);
            tail.prev.next = newone;
            tail.prev = newone;
        }
        else throw new IllegalArgumentException("Wrong!  addLast  !");
    }

    public Item removeFirst() {
        if (size != 0) {
            --size;
            Node old = head.next;
            head.next = old.next;
            old.next.prev = head;
            return old.data;
        }
        else throw new NoSuchElementException("Wrong!  removeFirst  !");
    }

    public Item removeLast() {
        if (size != 0) {
            --size;
            Node old = tail.prev;
            old.prev.next = tail;
            tail.prev = old.prev;
            return old.data;
        }
        else throw new NoSuchElementException("Wrong!  removeLast  !");
    }

    public Iterator<Item> iterator() {
        return new DequeIterator();
    }

    private class DequeIterator implements Iterator<Item> {
        private Node first = head.next;

        public boolean hasNext() {
            return first.next != null;
        }

        public Item next() {
            if (first.next != null) {
                Item item = first.data;
                first = first.next;
                return item;
            }
            else throw new NoSuchElementException("Wrong!  Iterator--next");
        }

        public void remove() {
            throw new UnsupportedOperationException("Wrong!  Iterator--remove");
        }

    }


    public static void main(String[] args) {
        Deque<Integer> dq = new Deque<Integer>();
        dq.addFirst(1);
        dq.addFirst(2);
        dq.addFirst(3);
        System.out.print("                                    size:");
        System.out.println(dq.size());
        dq.addFirst(4);
        dq.addFirst(5);
        dq.addFirst(6);
        dq.addFirst(7);

        System.out.println("test --- add  First---");
        System.out.print("                                    size:");
        System.out.println(dq.size());
        for (int item : dq) {
            System.out.println(item);
        }

        dq.addLast(10);
        dq.addLast(20);
        System.out.print("                                    size:");
        System.out.println(dq.size());
        dq.addLast(30);
        dq.addLast(40);

        System.out.print("size:");
        System.out.println(dq.size());
        System.out.println("test --- add  Last---");
        for (int item : dq) {
            System.out.println(item);
        }

        System.out.println();

        System.out.println("test --- remove  First---  1");
        System.out.println(dq.removeFirst());
        System.out.print("                                    size:");
        System.out.println(dq.size());
        System.out.println("test --- remove  First---  2");
        System.out.println(dq.removeFirst());
        System.out.print("                                    size:");
        System.out.println(dq.size());
        // System.out.println("--------------");

        System.out.println();

        System.out.println("test --- remove  Last---   1");
        System.out.println(dq.removeLast());
        System.out.println("test --- remove  Last---   2");
        System.out.println(dq.removeLast());
        System.out.print("                                    size:");
        System.out.println(dq.size());
        System.out.println("test --- remove  Last---   3");
        System.out.println(dq.removeLast());
        System.out.print("                                    size:");
        System.out.println(dq.size());
    }

}
