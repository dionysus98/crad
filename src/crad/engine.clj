(ns crad.engine
  (:require [crad.util :as util]
            [crad.db :as db]
            [crad.back-prop :refer [back-propgate]]))

(defn rid
  ([]
   (rid (->> (repeatedly 9 #(char (int (util/rrand 65 90))))
             (apply str)
             keyword)))
  ([id] (if (db/in-db id) (rid) id)))

(defn <v>
  [n & {grad :grad
        cn   :children
        op   :op}]
  (let [out {:id    (rid)
             :data  (double n)
             :grad  (or (some-> grad double) 0.0)
             :_prev cn
             :op    op}]
    (db/conj-db! out)
    out))

(defn v+ [a b]
  (<v> (+ (:data a) (:data b))
       :children [(:id a) (:id b)]
       :op :+))

(defn relu [{n :data :as a}]
  (<v> (if (< n 0) 0 n)
       :children [(:id a)]
       :op :relu))

(defn v- [a b]
  (<v> (- (:data a) (:data b))
       :children [(:id a) (:id b)]
       :op :-))

(defn v* [a b]
  (<v> (* (:data a) (:data b))
       :children [(:id a) (:id b)]
       :op :*))

(defn v** [a pow]
  (<v> (Math/pow (:data a) pow)
       :children [(:id a) pow]
       :op :**))

(defn vdiv [a b]
  (<v> (/ (:data a) (:data b))
       :children [(:id a) (:id b)]
       :op :/))

(defn backward [self]
  (let [bk- (fn back-prop! [s]          
              (let [s (back-propgate (db/get-by-id s))]
                (doseq [a (:_prev s)]
                  (back-prop! a))))]
    (db/update-db! (:id self) :grad 1.0)
    (bk- (:id self))))

(def a (<v> 5))
(def b (<v> 6))
(def c (<v> 7))
(def d (<v> 8))
(def e (vdiv c (v- (v+ d b) a)))
(def g (relu e))

(comment
  (db/reset-db!)
  [[(:data a) (:grad a)]
   [(:data b) (:grad b)]
   [(:data c) (:grad c)]
   [(:data d) (:grad d)]
   [(:data e) (:grad e)]
   [(:data g) (:grad g)]]

  (backward g)

  (db/get-by-id (:id b))

  {:id    :OPONUWCXF
   :data  6.0
   :grad  -0.08641975308641975
   :_prev nil
   :op    nil}
  
  [[5.0 0.0] [6.0 0.0] [7.0 0.0] [8.0 0.0] [0.7777777777777778 0.0] [0.7777777777777778 0.0]]

  :rcf)
