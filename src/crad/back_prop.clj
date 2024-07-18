(ns crad.back-prop
  (:require [crad.db :refer [update-db! get-by-id]]))

(defn relu [out]
  (let [[a]  (:_prev out)
        a    (get-by-id a)
        grad (+ (:grad a)
                (if (> (:data out) 0)
                  (:grad out)
                  0))]
    (update-db! (:id a) :grad grad)
    out))

(defn v+ [out]
  (let [[a b]  (:_prev out)
        a      (get-by-id a)
        b      (get-by-id b)
        grad-a (+ (:grad a) (:grad out))
        grad-b (+ (:grad b) (:grad out))]
    (update-db! (:id a) :grad grad-a)
    (update-db! (:id b) :grad grad-b)
    out))

(defn v- [out]
  (let [[a b]  (:_prev out)
        a      (get-by-id a)
        b      (get-by-id b)        
        grad-a (+ (:grad a) (:grad out))
        grad-b (- (:grad b) (:grad out))]
    (update-db! (:id a) :grad grad-a)
    (update-db! (:id b) :grad grad-b)
    out))

(defn v* [out]
  (let [[a b]  (:_prev out)
        a      (get-by-id a)
        b      (get-by-id b)
        grad-a (+ (:grad a) (* (:data b) (:grad out)))
        grad-b (- (:grad b) (* (:data a) (:grad out)))]
    (update-db! (:id a) :grad grad-a)
    (update-db! (:id b) :grad grad-b)
    out))

(defn v** [out]
  (let [[a pow] (:_prev out)
        a       (get-by-id a)
        grad-a  (+ (:grad a) (* (:grad out)
                                pow
                                (Math/pow (:data a) (dec pow))))]
    (update-db! (:id a) :grad grad-a)
    out))

(defn vdiv [out]
  (let [[a b] (:_prev out)
        a    (get-by-id a)
        b    (get-by-id b)
        grad-a (+ (:grad a) (* (Math/pow (:data b) -1.0) (:grad out)))
        grad-b (+ (:grad b) (* (/ (- (:data a))
                                  (Math/pow (:data b) 2.0))
                               (:grad out)))]
    (update-db! (:id a) :grad grad-a)
    (update-db! (:id b) :grad grad-b)
    out))

(defn back-propgate [out]
  (let [bp> (case (:op out)
              :+    v+
              :-    v-
              :*    v*
              :**   v**
              :/    vdiv
              :relu relu
              nil)]
    (if (fn? bp>) (bp> out) out)))
