(ns crad.nn
  (:require [crad.db :as db]
            [crad.engine :refer [<v>] :as e]
            [crad.nn :as nn]
            [crad.util :refer [rrand zip]]))

(def tes (atom nil))

(comment
  (e/backward @tes)
  :rcf)

(defn feed-forward-neuron [neuron xs]
  (let [!act (->> (zip (:w neuron) xs)
                  (mapv (fn [[!wi !xi]]
                          (e/v* (db/get-by-id (:id !wi))
                                (db/get-by-id (:id !xi)))))                  
                  (reduce (fn [cur nxt]
                            (e/v+ (db/get-by-id (:id cur))
                                  (db/get-by-id (:id nxt)))))
                  (e/v+ (db/get-by-id (:id (:b neuron)))))]
    (reset! tes (if (:nonlin? neuron)
      (e/relu !act)
      !act))))

(defn <neuron>
  [nin & {nonlin? :nonlin?
          :or     {nonlin? true}}]
  {:w       (mapv (fn [_] (<v> (rrand -1 1))) (range nin))
   :b       (<v> (rrand -1 1))
   :nonlin? nonlin?})

(defn feed-forward-layer [layer xs]
  (let [out (mapv (fn [neuron] (feed-forward-neuron neuron xs)) layer)]
    (case (count out)
      1 (first out)
      out)))

(defn <layer>
  [nin nout
   & {nonlin? :nonlin?
      :or     {nonlin? true}}]
  (mapv (fn [_] (<neuron> nin :nonlin? nonlin?)) (range nout)))

(defn feed-forward [model xs]
  (let [out (mapv (fn [layer] (feed-forward-layer layer xs)) model)]
    (last out)))

(defn <MLP> [nin nouts]
  (let [sz    (cons nin nouts)
        couts (count nouts)
        layer (fn [i]
                (<layer> (nth sz i) (nth sz (inc i))
                         :nonlin? (not= i couts)))]
    (mapv layer (range couts))))

(defn parameters [model]
  (vec
   (apply concat
          (for [layer model
                neuron layer]
            (conj (:w neuron) (:b neuron))))))

(defn clear! [model xs ys]
  (apply db/keep-items 
         (concat
          (map :id (nn/parameters model))
          (xs) ys)))